from collections import Iterable
from typing import List
from torch.utils.data import ConcatDataset

from Tasks.Task import Task
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
