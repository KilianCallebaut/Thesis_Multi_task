import os
import pickle
import random
import types

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import Dataset

from Tasks.Task import Task
import matplotlib.pyplot as plt
import librosa
import librosa.display

# In index mode:
# before save -> full tensors in input list
# after save or after to_index -> fragment id's in inputs
# when splitting into train and test -> filenames in inputs

class TaskDataset(Dataset):

    # data_lists: lists of lists of instances per task
    # now requires sampling to be called beforehand, save scalers to be called beforehand
    def __init__(
            self,
            inputs,
            targets,
            name,
            labels,
            extraction_method,
            base_path,
            output_module='softmax',  # 'sigmoid'
            index_mode=False,
            grouping=None
    ):
        if index_mode:
            self.switch_index_methods()
        else:
            self.scaled = False
        self.index_mode = index_mode
        self.inputs = inputs  # List of tensors
        self.targets = targets  # List of binary strings
        # self.name = name # String
        # self.labels = labels # List of strings
        self.task = Task(name, labels, output_module)
        self.extraction_method = extraction_method
        self.base_path = base_path
        self.pad_after = list()
        self.pad_before = list()
        self.grouping = grouping

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        if not self.scaled:
            self.inputs = self.extraction_method.scale_transform(self.inputs)
            self.scaled = True

        return self.inputs[index].float(), \
               torch.from_numpy(np.array(self.pad_before + self.targets[index] + self.pad_after)), \
               self.task.name

    def __len__(self):
        return len(self.inputs)

    def pad_targets(self, before: int, after: int):
        self.pad_before = [0 for _ in range(before)]
        self.pad_after = [0 for _ in range(after)]

    def save(self, base_path):
        joblib.dump(self.inputs, os.path.join(base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
        t_s = torch.Tensor(self.targets)
        torch.save(t_s, os.path.join(base_path, 'targets.pt'))

        diction = {
            'task': self.task,
            'pad_after': self.pad_after,
            'pad_before': self.pad_before
        }
        joblib.dump(diction, os.path.join(base_path, 'other.obj'))

    def load(self, base_path):
        self.inputs = joblib.load(os.path.join(base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
        t_l = torch.load(os.path.join(base_path, 'targets.pt'))
        self.targets = [[int(j) for j in i] for i in t_l]
        diction = joblib.load(os.path.join(base_path, 'other.obj'))
        self.task = diction['task']
        self.pad_after = diction['pad_after']
        self.pad_before = diction['pad_before']

    def sample_labels(self, dic_of_labels_limits, random_state=None):
        sampled_targets = self.targets
        sampled_inputs = self.inputs
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
        self.inputs = sampled_inputs
        self.targets = sampled_targets

    def sample_dataset(self, random_state=None):
        """
            Takes 1/5th of complete dataset
            :return: sampled datasets
        """
        it = self.k_folds(random_state=random_state)
        _, sample = next(it)
        return TaskDataset(
            inputs=[self.inputs[i] for i in sample],
            targets=[self.targets[i] for i in sample],
            name=self.task.name,
            labels=self.task.output_labels,
            extraction_method=self.extraction_method,
            base_path=self.base_path, output_module=self.task.output_module
        )

    # k_folds splitting
    def k_folds(self, random_state=None):
        # Examples
        # --------
        # >> > import numpy as np
        # >> > from sklearn.model_selection import KFold
        # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >> > y = np.array([1, 2, 3, 4])
        # >> > kf = KFold(n_splits=2)
        # >> > kf.get_n_splits(X)
        # 2
        # >> > print(kf)
        # KFold(n_splits=2, random_state=None, shuffle=False)
        # >> > for train_index, test_index in kf.split(X):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # ...
        # X_train, X_test = X[train_index], X[test_index]
        # ...
        # y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [2 3]
        # TEST: [0 1]
        # TRAIN: [0 1]
        # TEST: [2 3]

        # Stratified KFold
        # Examples
        # --------
        # >> > import numpy as np
        # >> > from sklearn.model_selection import StratifiedKFold
        # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >> > y = np.array([0, 0, 1, 1])
        # >> > skf = StratifiedKFold(n_splits=2)
        # >> > skf.get_n_splits(X, y)
        # 2
        # >> > print(skf)
        # StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
        # >> > for train_index, test_index in skf.split(X, y):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # ...
        # X_train, X_test = X[train_index], X[test_index]
        # ...
        # y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [1 3]
        # TEST: [0 2]
        # TRAIN: [0 2]
        # TEST: [1 3]

        # kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        # if dic_of_labels_limits is None:
        #     dic_of_labels_limits = {}
        # if dic_of_labels_limits:
        #     self.sample_labels(dic_of_labels_limits)
        inputs = self.inputs
        targets = self.targets

        if self.grouping:
            kf = GroupKFold(n_splits=5)
            return kf.split(inputs, groups=self.grouping)

        if self.task.output_module == 'softmax':
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            targets = LabelEncoder().fit_transform([''.join(str(l)) for l in targets])
            return kf.split(inputs, targets)
        else:
            kf = IterativeStratification(n_splits=5)
            targets = np.array(targets)
            return kf.split(inputs, targets)


    def get_split_by_index(self, train_index, test_index, **kwargs):
        '''

        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :param kwargs: extra arguments for loading scalers
        :return: the taskdataset for the training and test data
        '''

        x_train = [self.inputs[i] for i in train_index]
        y_train = [self.targets[i] for i in train_index]
        x_val = [self.inputs[i] for i in test_index]
        y_val = [self.targets[i] for i in test_index]

        if not self.extraction_method.scalers:
            self.extraction_method.scale_fit(x_train)

        train_taskdataset = TaskDataset(inputs=x_train, targets=y_train, name=self.task.name + "_train",
                                        labels=self.task.output_labels, extraction_method=self.extraction_method,
                                        base_path=self.base_path, output_module=self.task.output_module)
        test_taskdataset = TaskDataset(inputs=x_val, targets=y_val, name=self.task.name + "_test",
                                       labels=self.task.output_labels, extraction_method=self.extraction_method,
                                       base_path=self.base_path, output_module=self.task.output_module)
        return train_taskdataset, test_taskdataset

    # normal_mode method
    def save_split_scalers(self, random_state):
        i = 0
        for train, _ in self.k_folds(random_state):
            x_train = [self.inputs[ind] for ind in range(len(self.inputs)) if ind in train]
            self.extraction_method.scale_fit(x_train)
            scalers = self.extraction_method.scalers
            path = os.path.join(self.base_path,
                                'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                                  random_state, i))
            joblib.dump(value=scalers, filename=path, protocol=pickle.HIGHEST_PROTOCOL)
            i += 1

    def load_split_scalers(self, fold, random_state):
        path = os.path.join(self.base_path,
                            'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                              random_state, fold))
        self.extraction_method.scalers = joblib.load(path)

    def check_split_scalers(self, fold, random_state):
        path = os.path.join(self.base_path,
                            'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                              random_state, fold))
        return os.path.isfile(path)

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
        self.get_split_by_index = types.MethodType(get_split_by_index_index_mode, self)
        self.save = types.MethodType(save_index_mode, self)
        self.load = types.MethodType(load_index_mode, self)

    def has_index_mode(self):
        separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
        return os.path.isdir(separated_dir) and len(os.listdir(separated_dir)) == len(self.inputs)

    # frame input
    def prepare_inputs(self, **kwargs):
        self.inputs = self.extraction_method.prepare_inputs(self.inputs, **kwargs)

    @staticmethod
    def check(base_path, extraction_method):
        return os.path.isfile(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method))) and os.path.isfile(
            os.path.join(base_path, 'targets.pt')) and os.path.isfile(os.path.join(base_path, 'other.obj'))


def get_item_index_mode(self, index):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    separated_file = os.path.join(separated_dir, 'input_{}.pickle'.format(self.inputs[index]))
    x = self.extraction_method.scale_transform([torch.load(separated_file).float()])[0]
    y = self.targets[index]
    return x, \
           torch.from_numpy(np.array(self.pad_before + y + self.pad_after)), \
           self.task.name


def get_split_by_index_index_mode(self, train_index, test_index, **kwargs):
    '''
        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :param kwargs: extra arguments for loading scalers
        :return: the taskdataset for the training and test data
    '''

    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    x_train_window = train_index
    x_val_window = test_index

    y_train_window = [self.targets[i] for i in train_index]
    y_val_window = [self.targets[i] for i in test_index]

    if not self.extraction_method.scalers:
        print('calculating scalers')
        x_train = [torch.load(os.path.join(separated_dir, 'input_{}.pickle'.format(ind))).float()
                   for ind in x_train_window]
        self.extraction_method.scale_fit(x_train)

    train_taskdataset = TaskDataset(inputs=x_train_window, targets=y_train_window, name=self.task.name + "_train",
                                    labels=self.task.output_labels, extraction_method=self.extraction_method,
                                    base_path=self.base_path, output_module=self.task.output_module, index_mode=True)
    test_taskdataset = TaskDataset(inputs=x_val_window, targets=y_val_window, name=self.task.name + "_test",
                                   labels=self.task.output_labels, extraction_method=self.extraction_method,
                                   base_path=self.base_path, output_module=self.task.output_module, index_mode=True)

    return train_taskdataset, test_taskdataset


def save_index_mode(self, base_path):
    t_s = torch.Tensor(self.targets)
    torch.save(t_s, os.path.join(base_path, 'targets.pt'))

    diction = {
        'task': self.task,
        'pad_after': self.pad_after,
        'pad_before': self.pad_before
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
    self.pad_after = diction['pad_after']
    self.pad_before = diction['pad_before']

