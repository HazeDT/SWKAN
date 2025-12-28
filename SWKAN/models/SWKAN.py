import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.layers import trunc_normal_
from models.Efficient_KAN import KAN
import matplotlib.pyplot as plt
import os


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, alpha=0.1):
        super().__init__()
        fft_dim = dim // 2 + 1
        self.complex_weight_high = nn.Parameter(torch.randn(fft_dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(fft_dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, N = x_fft.shape

        real = x_fft.real
        imag = x_fft.imag
        epsilon = 1e-6
        magnitude = torch.sqrt(real.pow(2) + imag.pow(2) + epsilon)
        energy = magnitude.pow(2)

        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param

        return adaptive_mask

    def forward(self, x_in):
        if x_in.dim() == 2:
            x_expanded = x_in.unsqueeze(1)
        else:
            x_expanded = x_in
        B, C, N = x_expanded.shape
        x = x_expanded.view(B * C, N)
        x = x.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_masked = x_fft * freq_mask.to(x.device)
        weight_high = torch.view_as_complex(self.complex_weight_high)
        x_weighted2 = x_masked * weight_high
        x_weighted = x_weighted + x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = self.alpha * x.view(B, C, N) + (1 - self.alpha) * x_expanded
        # print('x',x.shape)
        return x


class WKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type):
        super(WKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.translation = nn.Parameter(torch.tensor(0.0))
        # self.scale = nn.Parameter(torch.ones(out_features, in_features))
        # self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        # print("wavelet_weight", self.wavelet_weights.shape)

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        # Base activation function
        self.base_activation = nn.SiLU()
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 3:
            x_expanded = x.squeeze(0)
        else:
            x_expanded = x
        self.translation_expanded = self.translation.expand(x_expanded.size(0))
        self.scale_expanded = self.scale.expand(x_expanded.size(0))
        # self.translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        # self.scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - self.translation_expanded[:, None]) / self.scale_expanded[:, None]
        # x_scaled = (x_expanded - self.translation_expanded) / self.scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = (1 - (x_scaled ** 2))
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'Laplace':
            A = 0.08
            ep = 0.03
            w = 2 * math.pi * 50
            indc = -ep / (torch.sqrt(torch.tensor(1 - pow(ep, 2))))
            wavelet = A * torch.exp(indc * w * x_scaled) * torch.sin(w * x_scaled)
            # wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            # wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'Dog':
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
        elif self.wavelet_type == 'Meyer':
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            wavelet = torch.sin(pi * v) * meyer_aux(v)
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet

    def forward(self, x):
        # Apply wavelet transform
        wavelet_output = self.wavelet_transform(x)
        return wavelet_output


class WKAN(nn.Module):
    def __init__(self, in_channel, out_channel, wavelet_type='mexican_hat'):
        super(WKAN, self).__init__()
        self.wavelet_type = wavelet_type

        self.adaptive_block1 = Adaptive_Spectral_Block(dim=1024)
        self.WKA1 = WKANLinear(1024, 1024, wavelet_type)

        self.adaptive_block2 = Adaptive_Spectral_Block(dim=1024)
        self.WKA2 = WKANLinear(1024, 1024, wavelet_type)
        self.KAN = KAN([3072, 512, out_channel])

    def forward(self, x):
        x = self.adaptive_block1(x)
        x1 = x.squeeze(1)
        out1 = self.WKA1(x1)
        x2 = self.adaptive_block2(out1)
        x3 = x2.squeeze(1)
        out2 = self.WKA2(x3)
        # residual_out1 = x1+out1+out3
        residual_out1 = torch.cat((x1, out1, out2), dim=1)
        out = self.KAN(residual_out1)
        # return out, out2
        return out
