import torch
import os
from torch.utils.data import Dataset
from scipy.io import loadmat

class RealDataset(Dataset):
    def __init__(self, real_dir):
        self.real_dir = real_dir
        self.real_signals = os.listdir(real_dir)

    def __len__(self):
        return len(self.real_signals)

    def __getitem__(self, idx):
        path_signal = os.path.join(self.real_dir, self.real_signals[idx])
        signal = torch.from_numpy(loadmat(path_signal)['data'])
        return signal


class SimulatedDataset(Dataset):
    def __init__(self, simulated_dir, list):
        self.simulated_dir = simulated_dir
        self.simulated_signals = list

    def __len__(self):
        return len(self.simulated_signals)

    def __getitem__(self, idx):
        path_mix = os.path.join(self.simulated_dir, self.simulated_signals[idx][0])
        path_mecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][1])
        path_fecg = os.path.join(self.simulated_dir, self.simulated_signals[idx][2])
        number_of_noise = self.simulated_signals[idx][3]
        number_snr = self.simulated_signals[idx][4]
        number_case = self.simulated_signals[idx][5]
        mix = torch.from_numpy(loadmat(path_mix)['data'])
        mecg = torch.from_numpy(loadmat(path_mecg)['data'])
        fecg = torch.from_numpy(loadmat(path_fecg)['data'])
        return mix, mecg, fecg, path_mix , path_mecg, path_fecg , number_of_noise, number_snr, number_case