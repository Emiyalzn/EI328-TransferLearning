import numpy as np
import scipy.io as sio
import os
from torch.utils.data import Dataset

DATASET_PATH = "../dataset"

def load_data():
    EEG_X = sio.loadmat(os.path.join(DATASET_PATH, "EEG_X.mat"))['X'][0]
    EEG_Y = sio.loadmat(os.path.join(DATASET_PATH, "EEG_Y.mat"))['Y'][0]
    return EEG_X, EEG_Y

def load_auxiliary_data():
    EEG_X = sio.loadmat(os.path.join(DATASET_PATH, "auxiliary_data_label_-1.mat"))['X']
    EEG_Y = sio.loadmat(os.path.join(DATASET_PATH, "auxiliary_data_label_-1.mat"))['Y'].transpose()
    for label in [0, 1]:
        new_X = sio.loadmat(os.path.join(DATASET_PATH, f"auxiliary_data_label_{label}.mat"))['X']
        new_Y = sio.loadmat(os.path.join(DATASET_PATH, f"auxiliary_data_label_{label}.mat"))['Y'].transpose()
        EEG_X = np.concatenate((EEG_X, new_X), axis=0)
        EEG_Y = np.concatenate((EEG_Y, new_Y), axis=0)
    return EEG_X, EEG_Y

class SeedDataset(Dataset):
    def __init__(self, is_train, is_augmentation=False):
        self.x_total, self.y_total = load_data()
        self.x, self.y = None, None
        self.is_train = is_train
        self.is_augmentation = is_augmentation
        if self.is_augmentation:
            self.x_auxiliary, self.y_auxiliary = load_auxiliary_data()

    def prepare_dataset(self, idx):
        if self.is_train:
            flag = 0
            for i in range(len(self.x_total)):
                if i == idx:
                    continue
                if flag == 0:
                    flag = 1
                    self.x = self.x_total[i]
                    self.y = self.y_total[i]
                else:
                    self.x = np.concatenate((self.x, self.x_total[i]), axis=0)
                    self.y = np.concatenate((self.y, self.y_total[i]), axis=0)
            if self.is_augmentation:
                self.x = np.concatenate((self.x, self.x_auxiliary), axis=0)
                self.y = np.concatenate((self.y, self.y_auxiliary), axis=0)
        else:
            self.x = self.x_total[idx]
            self.y = self.y_total[idx]

        self.x = np.array(self.x)
        self.y = np.squeeze(np.array(self.y))

    def prepare_gen(self):
        self.x = self.x_total[0]
        self.y = self.y_total[0]
        for i in range(len(self.x_total)):
            self.x = np.concatenate((self.x, self.x_total[i]), axis=0)
            self.y = np.concatenate((self.y, self.y_total[i]), axis=0)
        self.x = np.array(self.x)
        self.y = np.squeeze(np.array(self.y))

    def __getitem__(self, index):
        return self.x[index], self.y[index]+1

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    train_dataset = SeedDataset(False)
    train_dataset.prepare_dataset(1)
    print(train_dataset.y.shape)
