import os
import pickle
import random
import types
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification
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
            inputs: List[torch.tensor],
            targets: List[List],
            task: Task,
            extraction_method: ExtractionMethod,
            base_path: str,
            index_mode=False,
            grouping: List = None,
            extra_tasks: List[Tuple] = None
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
        if index_mode:
            self.switch_index_methods()
        self.index_mode = index_mode
        self.inputs = inputs  # List of tensors
        self.targets = targets  # List of binary strings
        # self.name = name # String
        # self.labels = labels # List of strings
        # self.task = Task(name=name, output_labels=labels, classification_type=classification_type,
        #                  loss_function=loss_function)
        self.task = task
        self.extraction_method = extraction_method
        self.base_path = base_path
        self.pad_after = list()
        self.pad_before = list()
        self.grouping = grouping
        self.extra_tasks = extra_tasks

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        return self.inputs[index].float(), \
               torch.from_numpy(np.array(self.pad_before + self.get_all_targets(index) + self.pad_after)), \
               self.task.name

    def get_all_targets(self, index):
        if not self.extra_tasks:
            return self.targets[index]
        targets = self.targets[index]
        for t in self.extra_tasks:
            targets += t[1][index]
        return targets

    def get_all_tasks(self):
        tasks = [self.task]
        if self.extra_tasks:
            tasks += [t[0] for t in self.extra_tasks]
        return tasks

    def __len__(self):
        return len(self.inputs)

    def pad_targets(self, before: int, after: int):
        self.pad_before = [0 for _ in range(before)]
        self.pad_after = [0 for _ in range(after)]

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
        self.inputs = joblib.load(os.path.join(self.base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
        t_l = torch.load(os.path.join(self.base_path, 'targets.pt'))
        self.targets = [[int(j) for j in i] for i in t_l]
        diction = joblib.load(os.path.join(self.base_path, 'other.obj'))
        self.task = diction['task']
        self.grouping = diction['grouping']
        if diction.keys().__contains__('extra_tasks'):
            self.extra_tasks = diction['extra_tasks']

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

    # # k_folds splitting
    # def k_folds(self, random_state=None, n_splits=5):
    #     """
    #     Produces a k_fold training/test split generator, depending on the task type
    #
    #     :param n_splits: number of folds
    #     :param random_state: optional int for reproducability purposes
    #     :return: _BaseKFold object
    #     """
    #     # Examples
    #     # --------
    #     # >> > import numpy as np
    #     # >> > from sklearn.model_selection import KFold
    #     # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    #     # >> > y = np.array([1, 2, 3, 4])
    #     # >> > kf = KFold(n_splits=2)
    #     # >> > kf.get_n_splits(X)
    #     # 2
    #     # >> > print(kf)
    #     # KFold(n_splits=2, random_state=None, shuffle=False)
    #     # >> > for train_index, test_index in kf.split(X):
    #     #     ...
    #     #     print("TRAIN:", train_index, "TEST:", test_index)
    #     # ...
    #     # X_train, X_test = X[train_index], X[test_index]
    #     # ...
    #     # y_train, y_test = y[train_index], y[test_index]
    #     # TRAIN: [2 3]
    #     # TEST: [0 1]
    #     # TRAIN: [0 1]
    #     # TEST: [2 3]
    #
    #     # Stratified KFold
    #     # Examples
    #     # --------
    #     # >> > import numpy as np
    #     # >> > from sklearn.model_selection import StratifiedKFold
    #     # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    #     # >> > y = np.array([0, 0, 1, 1])
    #     # >> > skf = StratifiedKFold(n_splits=2)
    #     # >> > skf.get_n_splits(X, y)
    #     # 2
    #     # >> > print(skf)
    #     # StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    #     # >> > for train_index, test_index in skf.split(X, y):
    #     #     ...
    #     #     print("TRAIN:", train_index, "TEST:", test_index)
    #     # ...
    #     # X_train, X_test = X[train_index], X[test_index]
    #     # ...
    #     # y_train, y_test = y[train_index], y[test_index]
    #     # TRAIN: [1 3]
    #     # TEST: [0 2]
    #     # TRAIN: [0 2]
    #     # TEST: [1 3]
    #
    #     # kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    #     # if dic_of_labels_limits is None:
    #     #     dic_of_labels_limits = {}
    #     # if dic_of_labels_limits:
    #     #     self.sample_labels(dic_of_labels_limits)
    #     inputs = self.inputs
    #     targets = self.targets
    #
    #     if self.grouping:
    #         kf = GroupKFold(n_splits=n_splits)
    #         return kf.split(inputs, groups=self.grouping)
    #
    #     if self.task.classification_type == 'multi-label':
    #         kf = IterativeStratification(n_splits=n_splits)
    #         targets = np.array(targets)
    #         return kf.split(inputs, targets)
    #     else:
    #         kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    #         targets = LabelEncoder().fit_transform([''.join(str(l)) for l in targets])
    #         return kf.split(inputs, targets)
    #
    # def get_split_by_index(self, train_index, test_index):
    #     '''
    #     Returns the training and test set from the given lists of indexes
    #
    #     :param train_index: indexes of the training set
    #     :param test_index: indexes of the test set
    #     :param kwargs: extra arguments for loading scalers
    #     :return: the taskdataset for the training and test data
    #     '''
    #
    #     x_train = [self.inputs[i] for i in train_index]
    #     y_train = [self.targets[i] for i in train_index]
    #     x_val = [self.inputs[i] for i in test_index]
    #     y_val = [self.targets[i] for i in test_index]
    #
    #     if not self.extraction_method.scalers:
    #         self.extraction_method.scale_fit(x_train)
    #
    #     train_taskdataset = TaskDataset(inputs=x_train, targets=y_train,
    #                                     task=self.task, extraction_method=self.extraction_method,
    #                                     base_path=self.base_path, grouping=[self.grouping[i] for i in train_index],
    #                                     extra_tasks=[(t[0], [t[1][targ_id] for targ_id in train_index]) for t in
    #                                                  self.extra_tasks])
    #     train_taskdataset.task.name = self.task.name + "_train"
    #     test_taskdataset = TaskDataset(inputs=x_val, targets=y_val,
    #                                    task=self.task,
    #                                    extraction_method=self.extraction_method,
    #                                    base_path=self.base_path, grouping=[self.grouping[i] for i in test_index],
    #                                    extra_tasks=[(t[0], [t[1][targ_id] for targ_id in test_index]) for t in
    #                                                 self.extra_tasks])
    #     test_taskdataset.task.name = self.task.name + "_test"
    #     return train_taskdataset, test_taskdataset

    # normal_mode method
    # def save_split_scalers(self, random_state):
    #     """
    #     Calculates and saves the scaler objects for normalization for each fold, given a random_state, in separate files
    #     :param random_state: optional int for reproducability purposes
    #     """
    #
    #     i = 0
    #     for train, _ in self.k_folds(random_state):
    #         x_train = [self.inputs[ind] for ind in range(len(self.inputs)) if ind in train]
    #         self.extraction_method.scale_fit(x_train)
    #         scalers = self.extraction_method.scalers
    #         path = os.path.join(self.base_path,
    #                             'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
    #                                                                               random_state, i))
    #         joblib.dump(value=scalers, filename=path, protocol=pickle.HIGHEST_PROTOCOL)
    #         i += 1
    #
    # def load_split_scalers(self, fold, random_state):
    #     """
    #     Loads the scaler objects for dataset normalization purposes for the specified fold and random state
    #     :param fold: The fold to load
    #     :param random_state: The random state the five fold datasets was created with
    #     """
    #     path = os.path.join(self.base_path,
    #                         'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
    #                                                                           random_state, fold))
    #     self.extraction_method.scalers = joblib.load(path)
    #
    # def check_split_scalers(self, fold, random_state):
    #     path = os.path.join(self.base_path,
    #                         'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
    #                                                                           random_state, fold))
    #     return os.path.isfile(path)

    def save_scalers(self):
        self.extraction_method.scale_fit(self.inputs)
        scalers = self.extraction_method.scalers
        path = os.path.join(self.base_path,
                            'scaler_method_{}.pickle'.format(self.extraction_method.name))
        joblib.dump(value=scalers, filename=path, protocol=pickle.HIGHEST_PROTOCOL)

    def load_scalers(self):
        path = os.path.join(self.base_path,
                            'scaler_method_{}.pickle'.format(self.extraction_method.name))
        self.extraction_method.scalers = joblib.load(path)

    def check_scalers(self):
        path = os.path.join(self.base_path,
                            'scaler_method_{}.pickle'.format(self.extraction_method.name))
        return os.path.isfile(path)

    # index mode transition
    def to_index_mode(self):
        # save inputs in separate files (if not done so already)
        self.index_mode = True
        # replace inputs with index lists
        self.inputs = [i for i in range(len(self.inputs))]
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
        self.get_item = types.MethodType(get_item_index_mode, self)
        self.save = types.MethodType(save_index_mode, self)
        self.load = types.MethodType(load_index_mode, self)

    def has_index_mode(self):
        separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
        return os.path.isdir(separated_dir) and len(os.listdir(separated_dir)) == len(self.inputs)

    # frame input
    def prepare_inputs(self, **kwargs):
        self.inputs = self.extraction_method.prepare_inputs(self.inputs, **kwargs)
        # self.inputs = np.array(self.inputs)

    def normalize_inputs(self):
        self.inputs = self.extraction_method.scale_transform(self.inputs)

    @staticmethod
    def check(base_path, extraction_method):
        return os.path.isfile(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method))) and os.path.isfile(
            os.path.join(base_path, 'targets.pt')) and os.path.isfile(os.path.join(base_path, 'other.obj'))


def get_item_index_mode(self, index):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    separated_file = os.path.join(separated_dir, 'input_{}.pickle'.format(self.inputs[index]))
    x = self.extraction_method.scale_transform([torch.load(separated_file).float()])[0]
    y = self.get_all_targets(index)
    return x, \
           torch.from_numpy(np.array(self.pad_before + y + self.pad_after)), \
           self.task.name


# def get_split_by_index_index_mode(self, train_index, test_index):
#     '''
#         :param train_index: indexes of the training set
#         :param test_index: indexes of the test set
#         :param kwargs: extra arguments for loading scalers
#         :return: the taskdataset for the training and test data
#     '''
#
#     separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
#     x_train_window = train_index
#     x_val_window = test_index
#
#     y_train_window = [self.targets[i] for i in train_index]
#     y_val_window = [self.targets[i] for i in test_index]
#
#     if not self.extraction_method.scalers:
#         print('calculating scalers')
#         x_train = [torch.load(os.path.join(separated_dir, 'input_{}.pickle'.format(ind))).float()
#                    for ind in x_train_window]
#         self.extraction_method.scale_fit(x_train)
#
#     train_taskdataset = TaskDataset(inputs=x_train_window, targets=y_train_window,
#                                     task=self.task,
#                                     extraction_method=self.extraction_method,
#                                     base_path=self.base_path, index_mode=True)
#     train_taskdataset.task.name = self.task.name + "_train"
#     test_taskdataset = TaskDataset(inputs=x_val_window, targets=y_val_window, task=self.task,
#                                    extraction_method=self.extraction_method,
#                                    base_path=self.base_path, index_mode=True)
#     test_taskdataset.task.name = self.task.name + "_test"
#
#     return train_taskdataset, test_taskdataset


def save_index_mode(self, base_path):
    t_s = torch.Tensor(self.targets)
    torch.save(t_s, os.path.join(base_path, 'targets.pt'))

    diction = {
        'task': self.task,
        'grouping': self.grouping,
        'extra_tasks': self.extra_tasks
    }
    joblib.dump(diction, os.path.join(base_path, 'other.obj'))


def load_index_mode(self, base_path):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    dir_list = os.listdir(separated_dir)
    self.inputs = [i for i in range(len(dir_list))]
    t_l = torch.load(os.path.join(base_path, 'targets.pt'))
    self.targets = [[int(j) for j in i] for i in t_l]
    diction = joblib.load(os.path.join(base_path, 'other.obj'))
    self.task = diction['task']
    self.grouping = diction['grouping']
    if diction.keys().__contains__('extra_tasks'):
        self.extra_tasks = diction['extra_tasks']
