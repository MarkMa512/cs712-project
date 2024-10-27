import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, file_name="train.npy", folder_path="./dataset", seq_len=14, candidate_len=3):
        super().__init__()
        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len - 1

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len - 1]  # exclude the last one
        next_point = self.data[(idx + 1) * self.seq_len - 1]

        labels = torch.tensor([3.0, 1, 0])

        return seq, cdd, next_point, labels


class TestDataset(Dataset):

    def __init__(self, file_name="public.npy", folder_path="./dataset", seq_len=13, candidate_len=3):
        super().__init__()
        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len]

        return seq, cdd


if __name__ == "__main__":
    dataset = TrainDataset()

    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for idx, (seq, cdd, next_point, labels) in enumerate(train_loader):
        print(seq.shape, cdd.shape)
        breakpoint()
