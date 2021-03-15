import numpy as np
import torch
from torch.utils.data.dataset import ConcatDataset


class TasksDataset(ConcatDataset):

    # data_lists: lists of lists of instances per task
    def __init__(
            self,
            data_lists,
            tasks,
            transform=None
    ):
        # form task level -  0: input data, 1: output labels - instance level
        self.data_lists = data_lists
        self.tasks = tasks
        self.transform = transform

    def __getitem__(self, index):
        list_idx = 0
        while index > len(self.data_lists[list_idx]) - 1:
            index -= len(self.data_lists[list_idx])
            list_idx += 1
        inputs = self.data_lists[list_idx][0][index]
        # inputs = torch.from_numpy(np.array(self.data_lists[list_idx][0][index]))
        labels = torch.from_numpy(np.array(self.data_lists[list_idx][1][index]))
        return inputs, labels, list_idx

    def __len__(self):
        return len([item for sublist in self.data_lists for item in sublist])
