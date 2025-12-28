import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.SequenceDatasets import dataset
from utils.sequence_aug import *
from tqdm import tqdm

signal_size = 1024

fault_name = ['H','I','O','B','C','0.5X_O','0.5X_I','0.5X_B','0.5X_C']
# fault_name = ['H', 'I', 'O', 'B', 'C']
# label
label = [i for i in range(9)]
# label = [i for i in range(5)]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data = []
    lab = []
    for i in tqdm(range(len(fault_name))):
        # WC = '_20Hz.xls'
        WC = '_VS_0_40_0Hz.xls'
        path2 = os.path.join('/tmp', root, fault_name[i] + WC)
        data1, lab1 = data_load(signal_size, filename=path2, label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


def data_load(signal_size, filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = pd.read_csv(filename, skiprows=range(22), header=None, sep='\t')[2]
    fl = fl.values
    fl = (fl - fl.min()) / (fl.max() - fl.min())  # 数据归一化处理
    # fl = fl.reshape(-1, )  # for KAN
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
    # while end <= fl[:signal_size * 1000].shape[0]:
        x = fl[start:end]
        data.append(x)
        lab.append(label)
        start += signal_size
        end += signal_size
    print(label, len(data))
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


class HUST_Bearing(object):
    num_classes = 9
    inputchannel = 1

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_prepare(self, test=False):
        list_data = get_files(self.data_dir, test)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.5, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train'))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val'))
            return train_dataset, val_dataset