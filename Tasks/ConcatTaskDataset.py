from typing import List

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from Tasks.TaskDataset import TaskDataset


class ConcatTaskDataset(ConcatDataset):
    """
    Concatenates different TaskDatasets, while padding the target vectors to allow for batches of mixed tasks and target
    sizes.

    """

    def __init__(self, datasets: List[TaskDataset]):
        for d in range(len(datasets)):
            before = 0
            after = 0
            if d > 0:
                for d_id in range(d):
                    all_tasks = datasets[d_id].get_all_tasks()
                    for t_id in range(all_tasks):
                        before += len(all_tasks[t_id].output_labels)
            if d < len(datasets) - 1:
                for d_id in range(d + 1, len(datasets)):
                    all_tasks = datasets[d_id].get_all_tasks()
                    for t_id in range(all_tasks):
                        after += len(all_tasks[t_id].output_labels)
            datasets[d].pad_targets(before, after)
        super().__init__(datasets)
        self.datasets = datasets

    @property
    def get_task_list(self):
        return [t for d in self.datasets for t in d.get_all_tasks()]

    @property
    def get_target_flags(self):
        """
        Gets the targets flags, meaning the list of which columns belong to which task when two or more tasks
        are present in the same batch
        :return: 2D list of booleans of size (total_nr_tasks, total_nr_labels) where True if the label index belongs to
         the task index
        """
        target_flags = []
        for d in self.datasets:
            all_tasks = d.get_all_tasks()
            for t_id in range(len(all_tasks)):
                before = 0
                after = 0
                if t_id > 0:
                    before = sum([len(all_tasks[i].output_labels) for i in range(t_id)])
                if t_id < len(all_tasks) - 1:
                    after = sum([len(all_tasks[i].output_labels) for i in range(t_id + 1, len(all_tasks))])
                task_padding = [False for _ in d.pad_before] + [False for _ in range(before)] + \
                               [True for _ in all_tasks[t_id].output_labels] + [False for _ in range(after)] + \
                               [False for _ in d.pad_after]
                target_flags.append(task_padding)
        return target_flags

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
