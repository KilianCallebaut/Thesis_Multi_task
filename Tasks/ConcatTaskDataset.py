from typing import List

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from Tasks.TaskDataset import TaskDataset


class ConcatTaskDataset(ConcatDataset):

    def __init__(self, datasets: List[TaskDataset]):
        for d in range(len(datasets)):
            before = 0
            after = 0
            if d > 0:
                before = sum([len(datasets[i].targets[0]) for i in range(d)])
            if d < len(datasets) - 1:
                after = sum([len(datasets[i].targets[0]) for i in range(d + 1, len(datasets))])
            datasets[d].pad_targets(before, after)
        super().__init__(datasets)
        self.datasets = datasets

    def get_task_list(self):
        return [d.task for d in self.datasets]


    def split_inputs_targets(self):
        inputs = torch.cat([torch.stack(d.inputs) for d in self.datasets]).float()
        targets = torch.cat([torch.stack(
            [torch.from_numpy(np.array(d.pad_before + t + d.pad_after)) for t in d.targets]
        ) for d in self.datasets])
        names = torch.tensor(
            [d_id for d_id in range(len(self.datasets)) for _ in range(self.datasets[d_id].__len__())])
        return inputs, targets, names

    @staticmethod
    def split_inputs_targets_static(datasets: List[TaskDataset]):
        inputs = torch.cat([torch.stack(d.inputs) for d in datasets]).float()
        targets = torch.cat([torch.stack(
            [torch.from_numpy(np.array(d.pad_before + t + d.pad_after)) for t in d.targets]
        ) for d in datasets])
        names = torch.tensor(
            [d_id for d_id in range(len(datasets)) for _ in range(datasets[d_id].__len__())])
        return inputs, targets, names