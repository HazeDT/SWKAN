import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.SequenceDatasets import dataset
from utils.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

datasetname = ['1ndBearing_ball', '1ndBearing_inner', '1ndBearing_mix(inner+outer+ball)', '1ndBearing_outer',
               '2ndPlanetary_brokentooth', '2ndPlanetary_missingtooth', '2ndPlanetary_normalstate',
               '2ndPlanetary_rootcracks', '2ndPlanetary_toothwear']

label = [i for i in range(9)]


def get_files(root):
    data = []
    lab = []

    for i in tqdm(range(len(datasetname))):
        data_name = 'Data_Chan1.txt'
        path2 = os.path.join('/tmp', root, datasetname[i], data_name)
        data1, lab1 = data_load(
            filename=path2,
            label=label[i]
        )
        data += data1
        lab += lab1
    return [data, lab]


def add_noise(x, snr):
    '''
    :param x: the raw siganl
    :param snr: the signal to noise ratio
    :return: noise signal
    '''
    d = np.random.randn(len(x))  # generate random noise
    P_signal = np.sum(abs(x) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (snr / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = x.reshape(-1) + noise
    return noise_signal


def data_load(filename, label):
    fl = pd.read_csv(filename, skiprows=range(14), header=None)
    fl = fl.values
    # fl = add_noise(fl, -5)
    fl = (fl - fl.min()) / (fl.max() - fl.min())
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size

    # Correct indentation here
    # while end <= fl[:signal_size * 1000].shape[0]:
    while end <= fl.shape[0]:
        x = fl[start:end]
        data.append(x)
        lab.append(label)
        # print(f"Assigned label: {lab}")
        start += signal_size
        end += signal_size

    return data, lab


def data_transforms(dataset_type="train"):
    transforms = {
        'train': Compose([
            Reshape(),
            # RandomAddGaussian(),
            # RandomScale(),
            # RandomStretch(),
            # RandomCrop(),
            Retype()
        ]),
        'val': Compose([
            Reshape(),
            Retype()
        ])
    }
    return transforms[dataset_type]


class XJTU_Gearbox(object):
    num_classes = 9
    inputchannel = 1

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_prepare(self, test=False):
        list_data = get_files(self.data_dir)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train'))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val'))
            return train_dataset, val_dataset