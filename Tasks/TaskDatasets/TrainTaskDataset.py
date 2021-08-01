import os
from typing import List, Tuple

import torch

from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset


class TrainTaskDataset(TaskDataset):

    def __init__(self,
                 extraction_method: ExtractionMethod,
                 base_path: str,
                 index_mode=False):
        super().__init__(extraction_method, base_path, index_mode)
        # task.name += '_train'

    def normalize_fit(self):
        self.extraction_method.scale_reset()
        for i in range(len(self.inputs)):
            self.extraction_method.partial_scale_fit(self.get_input(i))

    def default_base_path_extension(self):
        self.base_path = os.path.join(self.base_path, 'train_set')
