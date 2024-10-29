import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset_task1(Dataset):
    def __init__(self, file_name="train.npy", folder_path="./dataset", seq_len=14, candidate_len=3):
        super().__init__()

        label_set = [[0,1,2],[1,0,2],[2,1,0],[0,2,1],[1,2,0],[2,0,1]]
        orig_data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        data = np.zeros(shape = (orig_data.shape[0] * 6, orig_data.shape[1]))
        # print(data.shape)
        label = np.zeros(shape = (len(orig_data)//seq_len * 6, 3))
        flag = 0
        for idx in range(orig_data.shape[0]):
            if flag==90000:
                break
            data_ = orig_data[idx * seq_len: (idx+1) * seq_len]
            cdd_ = data_[10:13]
            for k in range(6):
                data[flag * seq_len: flag * seq_len + 10] = data_[:10]
                data[flag * seq_len + 10] = cdd_[label_set[k][0]]
                data[flag * seq_len + 11] = cdd_[label_set[k][1]]
                data[flag * seq_len + 12] = cdd_[label_set[k][2]]
                data[(flag + 1) * seq_len-1] = data_[-1] # data[flag * seq_len + 13]
                label[flag] = label_set[k]
                flag += 1
        # print(f"idx={idx}, flag={flag}")
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len - 1

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len - 1]  # exclude the last one
        next_point = self.data[(idx + 1) * self.seq_len - 1]
        labels = self.label[idx]
        return seq, cdd, next_point, labels


class TrainDataset_task2(Dataset):
    def __init__(self, file_name="train.npy", folder_path="./dataset", seq_len=14, candidate_len=3):
        super().__init__()
        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len - 1

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        seq = self.data[idx * self.seq_len : (idx + 1) * self.seq_len - 1] # the first 13 datapoints
        next_point = self.data[(idx + 1) * self.seq_len - 1] # the last datapoint
        return seq, next_point

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


# class TrainDataset(Dataset):
#     def __init__(self, file_name="train.npy", folder_path="./dataset", seq_len=14, candidate_len=3):
#         super().__init__()
#         self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
#         self.seq_len = seq_len
#         self.candidate_len = candidate_len
#         self.given_seq_len = seq_len - candidate_len - 1

#     def __len__(self):
#         return len(self.data) // self.seq_len

#     def __getitem__(self, idx):
#         seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
#         cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len - 1]  # exclude the last one
#         next_point = self.data[(idx + 1) * self.seq_len - 1]

#         # zzh: Why not [2,1,0]? Why [3,1,0]?
#         labels = torch.tensor([3.0, 1, 0])

#         return seq, cdd, next_point, labels