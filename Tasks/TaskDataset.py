import os
import random
import types
from typing import List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.Task import Task


# In index mode:
# before save -> full tensors in input list
# after save or after to_index -> fragment id's in inputs
# when splitting into train and test -> filenames in inputs

class TaskDataset(Dataset):
    """
    Unified object to store dataset and task information and perform transformations.

    Has 2 modes: index and non-index mode.
    In index mode, the object only stores indices pointing to input files. These should be created first using the
    write_index_files method. When indexing, the inputs are live read from the corresponding file.
    In non-index mode, the whole dataset is loaded into memory.
    """

    # data_lists: lists of lists of instances per task
    # now requires sampling to be called beforehand, save scalers to be called beforehand
    def __init__(
            self,
            extraction_method: ExtractionMethod,
            base_path: str,
            index_mode=False
    ):
        """
        Initializes the Taskdataset object

        :param inputs: list of tensors of size (n_samples, n_features)
        :param targets: list of binary representing the targets
        :param task: Task object containing the name, list of distinct output labels and possible additional parameters
        :param extraction_method: ExtractionMethod object, responsible for appropriate transformations and name of the data for saving and loading
        :param base_path: The path to the folder where the data should be saved
        :param index_mode: Boolean to switch on/off index mode
        :param grouping: optional Grouping list for defining which data cannot be split up in k folds (see sklearn.model_selection.GroupKfold)
        :param extra_tasks: List of Tuples (Task, targets) holding additional tasks on the same inputs
        """
        self.index_mode = index_mode
        self.extraction_method = extraction_method
        self.base_path = base_path
        self.start_index_list = []
        self.stop_index_list = []
        self.total_target_size = 0

        self.prepared = False
        self.flag_scaled = False

        self.inputs = []
        self.targets = []
        self.grouping = []
        self.extra_tasks = []

    ########################################################################################################
    # Setters
    ########################################################################################################
    def initialize(self, inputs: List[torch.tensor], targets: List[List[int]], task: Task, grouping: List[int] = None,
                   extra_tasks: List[Tuple[Task, List[List[int]]]] = None):
        assert len(inputs) == len(targets), 'There must be as many inputs as there are targets'
        if grouping:
            assert len(grouping) == len(inputs), 'The grouping must contain as many elements as the inputs'
        if extra_tasks:
            for t in extra_tasks:
                assert len(t[1]) == len(inputs), 'Task {} does not have as many targets as there are inputs'.format(
                    t[0].name)
        self.inputs = inputs  # List of tensors
        self.targets = targets  # List of binary strings
        self.task = task
        self.grouping = grouping
        self.extra_tasks = extra_tasks

        for t in self.get_all_tasks():
            self.start_index_list += [self.total_target_size]
            self.total_target_size += len(t.output_labels)
            self.stop_index_list += [self.total_target_size]

        if self.index_mode:
            self.write_index_files()
            self.to_index_mode()

    ########################################################################################################
    # Getters
    ########################################################################################################

    def __getitem__(self, index):
        return self.extraction_method.scale_transform(self.get_input(index)), \
               torch.from_numpy(self.get_all_targets(index)), \
               torch.tensor([t.task_group for t in self.get_all_tasks()])

    def get_input(self, index):
        return self.inputs[index].float()

    def get_all_targets(self, index):
        targets = np.zeros(self.total_target_size, dtype=int)
        targets[self.start_index_list[0]:self.stop_index_list[0]] = self.targets[index]
        for i in range(len(self.extra_tasks)):
            targets[self.start_index_list[i+1]:self.stop_index_list[i+1]] = self.extra_tasks[i][1][index]
        return targets

    def get_all_tasks(self):
        tasks = [self.task]
        if self.extra_tasks:
            tasks += [t[0] for t in self.extra_tasks]
        return tasks

    def __len__(self):
        return len(self.inputs)

    ########################################################################################################
    # Combining
    ########################################################################################################
    def pad_targets(self, start_index_list: List[int], stop_index_list: List[int], total_size: int):
        self.start_index_list = start_index_list
        self.stop_index_list = stop_index_list
        self.total_target_size = total_size

    ########################################################################################################
    # Filtering
    ########################################################################################################

    def sample_labels(self, dic_of_labels_limits, random_state=None):
        """
        Samples the instances according to predefined limits of number of instances per label
        :param dic_of_labels_limits: Dictionary specifying the maximum amount of instances a label can have in the dataset
        :param random_state: optional int for reproducability purposes
        """
        sampled_targets = self.targets
        sampled_inputs = self.inputs
        sampled_grouping = self.grouping
        if random_state is not None:
            random.seed(random_state)

        for l in dic_of_labels_limits.keys():

            label_set = [i for i in range(len(sampled_targets))
                         if sampled_targets[i][self.task.output_labels.index(l)] == 1]
            if len(label_set) > dic_of_labels_limits[l]:
                random_label_set = random.sample(label_set, dic_of_labels_limits[l])
                sampled_targets = [sampled_targets[i] for i in range(len(sampled_targets)) if
                                   (i not in label_set or i in random_label_set)]
                sampled_inputs = [sampled_inputs[i] for i in range(len(sampled_inputs)) if
                                  (i not in label_set or i in random_label_set)]
                if self.grouping:
                    sampled_grouping = [sampled_grouping[i] for i in range(len(sampled_grouping)) if
                                        (i not in label_set or i in random_label_set)]
        self.inputs = sampled_inputs
        self.targets = sampled_targets
        self.grouping = sampled_grouping

    ########################################################################################################
    # Transformation
    ########################################################################################################

    def prepare_fit(self):
        for i in range(len(self.inputs)):
            self.extraction_method.prepare_fit(self.get_input(i))

    def prepare_inputs(self):
        if not self.grouping:
            self.grouping = [i for i in range(len(self.inputs))]
        for i in range(len(self.inputs)):
            prepared_inputs = self.extraction_method.prepare_input(self.get_input(i))
            self.inputs[i] = prepared_inputs[0]
            for inp_id in range(1, len(prepared_inputs)):
                self.inputs.append(prepared_inputs[inp_id])
                self.targets.append(self.targets[i])
                self.grouping.append(self.grouping[i])
                for t in self.extra_tasks:
                    t[1].append(t[1][i])

        self.prepared = True

    ########################################################################################################
    # Index mode
    ########################################################################################################

    def to_index_mode(self):
        # save inputs in separate files (if not done so already)
        self.index_mode = True
        # replace inputs with index lists
        self.inputs = [(i, 0) for i in range(len(self.inputs))]
        self.switch_index_methods()

    def write_index_files(self):
        separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
        if not os.path.exists(separated_dir):
            os.mkdir(separated_dir)

        if not os.listdir(separated_dir):
            for i in range(len(self.inputs)):
                input_path = os.path.join(separated_dir, 'input_{}.pickle'.format(i))
                torch.save(self.inputs[i], input_path)

    def switch_index_methods(self):
        # replace getitem, get_split_by_index by index based functions
        self.get_input = types.MethodType(get_input_index_mode, self)
        self.save = types.MethodType(save_index_mode, self)
        self.load = types.MethodType(load_index_mode, self)
        self.prepare_inputs = types.MethodType(prepare_inputs_index_mode, self)

    def has_index_mode(self):
        separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
        return os.path.isdir(separated_dir) and len(os.listdir(separated_dir)) == len(self.inputs)

    ########################################################################################################
    # I/O
    ########################################################################################################
    def save(self):
        joblib.dump(self.inputs, os.path.join(self.base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
        t_s = torch.Tensor(self.targets)
        torch.save(t_s, os.path.join(self.base_path, 'targets.pt'))

        diction = {
            'task': self.task,
            'grouping': self.grouping,
            'extra_tasks': self.extra_tasks
        }
        joblib.dump(diction, os.path.join(self.base_path, 'other.obj'))

    def load(self):
        inputs = joblib.load(os.path.join(self.base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
        t_l = torch.load(os.path.join(self.base_path, 'targets.pt'))
        t_l = [[int(j) for j in i] for i in t_l]
        diction = joblib.load(os.path.join(self.base_path, 'other.obj'))
        task = diction['task']
        grouping = diction['grouping']
        extra_tasks = diction['extra_tasks']
        self.initialize(inputs=inputs, targets=t_l, task=task, grouping=grouping, extra_tasks=extra_tasks)

    @staticmethod
    def check(base_path: str, extraction_method: ExtractionMethod):
        return os.path.isfile(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method.name))) \
               and os.path.isfile(os.path.join(base_path, 'targets.pt')) \
               and os.path.isfile(os.path.join(base_path, 'other.obj'))


def get_input_index_mode(self, index):
    separated_file = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name),
                                  'input_{}.pickle'.format(self.inputs[index][0]))
    input_tensor = torch.load(separated_file).float()
    if self.prepared:
        input_tensor = self.extraction_method.prepare_input(input_tensor)[self.inputs[index][1]]
    return input_tensor


def save_index_mode(self):
    t_s = torch.Tensor([self.targets[t] for t in range(len(self.targets)) if self.inputs[t][1] == 0])
    torch.save(t_s, os.path.join(self.base_path, 'targets.pt'))

    diction = {
        'task': self.task,
        'grouping': self.grouping,
        'extra_tasks': self.extra_tasks
    }
    joblib.dump(diction, os.path.join(self.base_path, 'other.obj'))


def load_index_mode(self):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    dir_list = os.listdir(separated_dir)
    self.inputs = [(i, 0) for i in range(len(dir_list))]
    t_l = torch.load(os.path.join(self.base_path, 'targets.pt'))
    self.targets = [[int(j) for j in i] for i in t_l]
    diction = joblib.load(os.path.join(self.base_path, 'other.obj'))
    self.task = diction['task']
    self.grouping = diction['grouping']
    if diction.keys().__contains__('extra_tasks'):
        self.extra_tasks = diction['extra_tasks']


def prepare_inputs_index_mode(self):
    if not self.grouping:
        self.grouping = [i[0] for i in self.inputs]
    for i in range(len(self.inputs)):
        prep_inputs = self.extraction_method.prepare_input(self.get_input(i))
        for j in range(1, len(prep_inputs)):
            self.inputs.append((i, j))
            self.targets.append(self.targets[i])
            self.grouping.append(self.grouping[i])
            for t in self.extra_tasks:
                t[1].append(t[1][i])
    self.prepared = True
