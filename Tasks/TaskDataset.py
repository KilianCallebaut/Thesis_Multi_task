import os

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from Tasks.Task import Task


class TaskDataset(Dataset):

    # data_lists: lists of lists of instances per task
    def __init__(
            self,
            inputs,
            targets,
            name,
            labels,
            output_module='softmax'  # 'sigmoid'
    ):
        self.inputs = inputs  # List of tensors
        self.targets = targets  # List of binary strings
        # self.name = name # String
        # self.labels = labels # List of strings
        self.task = Task(name, labels, output_module)
        self.pad_after = list()
        self.pad_before = list()

    def __getitem__(self, index):
        return self.inputs[index].float(), \
               torch.from_numpy(np.array(self.pad_before + self.targets[index] + self.pad_after)), \
               self.task.name

    def __len__(self):
        return len(self.inputs)

    def pad_targets(self, before: int, after: int):
        self.pad_before = [0 for _ in range(before)]
        self.pad_after = [0 for _ in range(after)]

    def save(self, base_path, extraction_method):
        joblib.dump(self.inputs, os.path.join(base_path, '{}_inputs.gz'.format(extraction_method)))
        t_s = torch.Tensor(self.targets)
        torch.save(t_s, os.path.join(base_path, 'targets.pt'))

        diction = {
            'task': self.task,
            'pad_after': self.pad_after,
            'pad_before': self.pad_before
        }
        joblib.dump(diction, os.path.join(base_path, 'other.obj'))

    def load(self, base_path, extraction_method):
        self.inputs = joblib.load(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method)))
        t_l = torch.load(os.path.join(base_path, 'targets.pt'))
        self.targets = [[int(j) for j in i] for i in t_l]
        diction = joblib.load(os.path.join(base_path, 'other.obj'))
        self.task = diction['task']
        self.pad_after = diction['pad_after']
        self.pad_before = diction['pad_before']

    @staticmethod
    def check(base_path, extraction_method):
        return os.path.isfile(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method))) and os.path.isfile(
            os.path.join(base_path, 'targets.pt')) and os.path.isfile(os.path.join(base_path, 'other.obj'))
