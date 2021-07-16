import os
from typing import List, Tuple

import torch

from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset


class TestTaskDataset(TaskDataset):

    def __init__(self,
                 inputs: List[torch.tensor],
                 targets: List[List],
                 task: Task,
                 extraction_method: ExtractionMethod,
                 base_path: str,
                 index_mode=False,
                 grouping: List = None,
                 extra_tasks: List[Tuple] = None):
        super().__init__(inputs, targets, task, extraction_method, base_path, index_mode, grouping, extra_tasks)
        # task.name += '_test'

    def default_base_path_extension(self):
        self.base_path = os.path.join(self.base_path, 'test_set')