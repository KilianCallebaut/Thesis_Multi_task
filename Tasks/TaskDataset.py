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

        :param extraction_method: ExtractionMethod object, responsible for appropriate transformations and name of the data for saving and loading
        :param base_path: The path to the folder where the data should be saved
        :param index_mode: Boolean to switch on/off index mode
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
        self.task = None
        self.grouping = []
        self.extra_tasks = []

        if self.index_mode:
            self.switch_index_methods()
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)

    ########################################################################################################
    # Initialization
    ########################################################################################################

    def add_input(self, input_tensor: torch.tensor):
        """
        Adds an input tensor to the dataset
        :param input_tensor: The feature matrix to insert into the dataset
        """
        self.inputs.append(input_tensor)

    def extract_and_add_input(self,
                              sig_samplerate: Tuple[np.ndarray, int]):
        """
        Adds an input tensor to the dataset
        :param sig_samplerate: The tuple with the signal object and the samplerate
        """

        if len(sig_samplerate[0].shape) > 1:
            channel = 0 if sig_samplerate[0].shape[0] < sig_samplerate[0].shape[1] else 1
            input_tensor = torch.tensor(
                [self.extraction_method.extract_features((sig_samplerate[0][:, i], sig_samplerate[channel])) for i in
                 range(sig_samplerate[0].shape[channel])])
        else:
            input_tensor = self.extraction_method.extract_features(sig_samplerate)
        self.add_input(input_tensor)

    def add_task_and_targets(self, task: Task, targets: List[List[int]]):
        """
        Adds a task object with a list of targets to the dataset.
        The targets should be in the same order as their corresponding inputs
        :param task: The task object to add
        :param targets: The list of target vectors to add
        """
        assert len(targets) == len(self.inputs), 'There must be as many targets as there are inputs'
        for t in targets:
            assert len(t) == len(targets[0]), 'Each target vector must have the same length'
        if not self.targets:
            self.targets = targets
            self.task = task
        else:
            self.extra_tasks.append((task, targets))

        self.start_index_list += [self.total_target_size]
        self.total_target_size += len(task.output_labels)
        self.stop_index_list += [self.total_target_size]

    def add_grouping(self, grouping: List[int]):
        """
        Adds the grouping list.
        The groupings should be in the same order as their corresponding inputs
        :param grouping: optional Grouping list for defining which data cannot be split up in k folds (see sklearn.model_selection.GroupKfold)
        """
        assert len(grouping) == len(self.inputs), 'There must be as many elements in the grouping as there are inputs'
        self.grouping = grouping

    def validate(self):
        """
        Checks if the object is valid
        :return: Bool
        """
        assert len(self.targets) == len(self.inputs), 'There must be as many targets as there are inputs'
        for t in self.targets:
            assert len(t) == len(self.targets[0]), 'Each target vector must have the same length'
        for t_e in self.extra_tasks:
            for t in t_e[1]:
                assert len(t) == len(t_e[1][0]), 'Each target vector must have the same length'
        if self.grouping:
            assert len(self.grouping) == len(
                self.inputs), 'There must be as many elements in the grouping as there are inputs'

        if self.inputs and self.targets and self.task:
            return True
        return False

    ########################################################################################################
    # Getters
    ########################################################################################################

    def __getitem__(self, index):
        """
        The getter method for returning inputs for the data loading of the training
        :param index: The index of the data instance to get
        :return: The feature matrix, the (combined) target vector and the task group
        """
        return self.get_input(index), \
               torch.from_numpy(self.get_all_targets(index)), \
               torch.tensor([t.task_group for t in self.get_all_tasks()])

    def get_input(self, index):
        if self.flag_scaled:
            return self.extraction_method.scale_transform(self.inputs[index]).float()
        return self.inputs[index].float()

    def get_all_targets(self, index):
        targets = np.zeros(self.total_target_size, dtype=int)
        targets[self.start_index_list[0]:self.stop_index_list[0]] = self.targets[index]
        for i in range(len(self.extra_tasks)):
            targets[self.start_index_list[i + 1]:self.stop_index_list[i + 1]] = self.extra_tasks[i][1][index]
        return targets

    def get_all_tasks(self):
        tasks = [self.task]
        if self.extra_tasks:
            tasks += [t[0] for t in self.extra_tasks]
        return tasks

    def copy_non_data_variables(self, taskDataset):
        self.start_index_list = taskDataset.start_index_list
        self.stop_index_list = taskDataset.stop_index_list
        self.total_target_size = taskDataset.total_target_size

        self.prepared = taskDataset.prepared
        self.flag_scaled = taskDataset.flag_scaled

    def __len__(self):
        return len(self.inputs)

    ########################################################################################################
    # Combining
    ########################################################################################################
    def pad_targets(self, start_index_list: List[int], stop_index_list: List[int], total_size: int):
        """
        Store the necessary data to combine the targets from each tasks with others
        :param start_index_list: List of starting indexes per task for the combined target vector
        :param stop_index_list: List of ending indexes per task for the combined target vector
        :param total_size: Total size of the combined target vector
        """
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
        sampled_extra_tasks = self.extra_tasks
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
                if self.extra_tasks:
                    for t in sampled_extra_tasks:
                        t[1] = [t[1][i] for i in range(len(t[1])) if
                                (i not in label_set or i in random_label_set)]
        self.inputs = sampled_inputs
        self.targets = sampled_targets
        self.grouping = sampled_grouping
        self.extra_tasks = sampled_extra_tasks

    ########################################################################################################
    # Transformation
    ########################################################################################################

    def normalize_fit(self):
        """
        Calculates the necessary metrics on the datset to use later for scaling the inputs
        """
        print("Calculate Normalize Fit")

        self.extraction_method.scale_reset()
        self.flag_scaled = False
        for i in range(len(self.inputs)):
            self.extraction_method.partial_scale_fit(self.get_input(i))

        self.flag_scaled = True

    def prepare_fit(self):
        """
        Calculates the necessary metrics on the dataset to use later for preparation
        """
        print("Calculate Preparation Fit")

        self.extraction_method.prepare_reset()
        self.prepared = False
        for i in range(len(self.inputs)):
            self.extraction_method.prepare_fit(self.get_input(i))

    def prepare_inputs(self):
        """
        Applies preparation of the input instances defined in the extraction method
        on each input instance in the TaskDataset.
        """
        self.prepared = False
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

    def switch_index_methods(self):
        # replace getitem, get_split_by_index by index based functions
        self.add_input = types.MethodType(add_input_index_mode, self)
        self.get_input = types.MethodType(get_input_index_mode, self)
        self.save = types.MethodType(save_index_mode, self)
        self.load_inputs = types.MethodType(load_inputs_index_mode, self)
        self.prepare_inputs = types.MethodType(prepare_inputs_index_mode, self)

    ########################################################################################################
    # I/O
    ########################################################################################################
    def save(self):
        joblib.dump(self.inputs, os.path.join(self.base_path, '{}_inputs.gz'.format(self.extraction_method.name)))

        diction = {
            'task': self.task,
            'targets': self.targets,
            'grouping': self.grouping,
            'extra_tasks': self.extra_tasks
        }
        joblib.dump(diction, os.path.join(self.base_path, 'task_info.obj'))

        diction = {
            'extraction_method': self.extraction_method
        }
        joblib.dump(diction,
                    os.path.join(self.base_path, '{}_extraction_method_params'.format(self.extraction_method.name)))

    def load(self):
        self.load_inputs()
        diction = joblib.load(os.path.join(self.base_path, 'task_info.obj'))
        self.task = diction['task']
        self.targets = diction['targets']
        self.grouping = diction['grouping']
        self.extra_tasks = diction['extra_tasks']

        diction = joblib.load(
            os.path.join(self.base_path, '{}_extraction_method_params'.format(self.extraction_method.name)))
        self.extraction_method.__dict__.update(diction['extraction_method'].__dict__)

    def load_inputs(self):
        self.inputs = joblib.load(os.path.join(self.base_path, '{}_inputs.gz'.format(self.extraction_method.name)))

    def check(self):
        check = os.path.isfile(os.path.join(self.base_path, 'task_info.obj')) \
                and os.path.isfile(
            os.path.join(self.base_path, '{}_extraction_method_params'.format(self.extraction_method.name)))

        if self.index_mode:
            return os.path.isdir(os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))) \
                   and check
        else:
            return os.path.isfile(os.path.join(self.base_path, '{}_inputs.gz'.format(self.extraction_method.name))) \
                   and check


########################################################################################################
# Initialization
########################################################################################################

def add_input_index_mode(self, input_tensor: torch.tensor):
    """
    Adds an input tensor to the dataset
    :param input_tensor:
    """
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    if not os.path.exists(separated_dir):
        os.mkdir(separated_dir)

    input_path = os.path.join(separated_dir, 'input_{}.pickle'.format(len(self.inputs)))
    torch.save(input_tensor, input_path)
    self.inputs.append((len(self.inputs), 0))


########################################################################################################
# Getters
########################################################################################################

def get_input_index_mode(self, index):
    separated_file = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name),
                                  'input_{}.pickle'.format(self.inputs[index][0]))
    input_tensor = torch.load(separated_file)
    if self.prepared:
        input_tensor = self.extraction_method.prepare_input(input_tensor)[self.inputs[index][1]]
    if self.flag_scaled:
        input_tensor = self.extraction_method.scale_transform(input_tensor)
    return input_tensor.float()


########################################################################################################
# Transformation
########################################################################################################

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


########################################################################################################
# I/O
########################################################################################################

def save_index_mode(self):
    diction = {
        'task': self.task,
        'targets': self.targets,
        'grouping': self.grouping,
        'extra_tasks': self.extra_tasks
    }
    joblib.dump(diction, os.path.join(self.base_path, 'task_info.obj'))

    diction = {
        'extraction_method': self.extraction_method
    }
    joblib.dump(diction,
                os.path.join(self.base_path, '{}_extraction_method_params'.format(self.extraction_method.name)))


def load_inputs_index_mode(self):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    dir_list = os.listdir(separated_dir)
    self.inputs = [(i, 0) for i in range(len(dir_list))]
