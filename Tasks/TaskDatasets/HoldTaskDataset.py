import os
import pickle
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification

from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset
from Tasks.TaskDatasets.TestTaskDataset import TestTaskDataset
from Tasks.TaskDatasets.TrainTaskDataset import TrainTaskDataset


class HoldTaskDataset(TaskDataset):
    """
    The TaskDataset object that unifies and splits the training/test sets
    """

    def __init__(self,
                 inputs: List[torch.tensor],
                 targets: List[List],
                 task: Task,
                 extraction_method: ExtractionMethod,
                 base_path: str = '',
                 index_mode=False,
                 grouping: List[int] = None,
                 extra_tasks: List[Tuple] = None,
                 training_base_path: str = None,
                 testing_base_path: str = None):
        super().__init__(inputs, targets, task, extraction_method, base_path, index_mode, grouping, extra_tasks)
        # self.training_indexes = []
        # self.test_indexes = []
        self.training_set = None
        self.test_set = None
        self.training_set = TrainTaskDataset(
            inputs=[],
            targets=[],
            task=self.task,
            extraction_method=self.extraction_method,
            base_path=training_base_path if training_base_path else os.path.join(self.base_path, 'train_set'),
            index_mode=self.index_mode,
            grouping=[],
            extra_tasks=[]
        )
        self.test_set = TestTaskDataset(
            inputs=[],
            targets=[],
            task=self.task,
            extraction_method=self.extraction_method,
            base_path=testing_base_path if testing_base_path else os.path.join(self.base_path, 'test_set'),
            index_mode=self.index_mode,
            grouping=[],
            extra_tasks=[]
        )

    def save(self):
        if self.check_train_test_present():
            self.training_set.save()
            self.test_set.save()
        else:
            super().save()

    def load(self):
        if os.path.isdir(self.training_set.base_path):
            self.training_set.load()
            self.test_set.load()
            self.task = self.training_set.task
            self.task.name = self.training_set.task.name.split('_train')[0]
            if self.training_set.grouping and self.test_set.grouping:
                self.grouping = self.training_set.grouping + [t + len(self.training_set.grouping) for t in
                                                              self.test_set.grouping]
            if self.training_set.extra_tasks and self.test_set.extra_tasks:
                self.extra_tasks = [(self.training_set.extra_tasks[t_id][0],
                                     self.training_set.extra_tasks[t_id][1] + self.test_set.extra_tasks[t_id][1]) for
                                    t_id in range(len(self.training_set.extra_tasks))]
        else:
            super().load()

    def check_train_test_present(self):
        return len(self.training_set.inputs) > 0 and len(self.test_set.inputs) > 0

    def add_train_test_set(self, training_inputs: List[torch.tensor], training_targets: List[List],
                           testing_inputs: List[torch.tensor], testing_targets: List[List],
                           training_grouping: List[int] = None, training_extra_tasks: List[Tuple] = None,
                           testing_grouping: List[int] = None, testing_extra_tasks: List[Tuple] = None):

        self.training_set.inputs = training_inputs
        self.training_set.targets = training_targets
        self.training_set.grouping = training_grouping
        self.training_set.extra_tasks = training_extra_tasks
        # self.training_indexes = [i for i in range(len(self.training_set))]

        self.test_set.inputs = testing_inputs
        self.test_set.targets = testing_targets
        self.test_set.grouping = testing_grouping
        self.test_set.extra_tasks = testing_extra_tasks
        # self.test_indexes = [i + len(self.training_set) for i in range(len(self.test_set))]

        self.inputs = [None for i in range(len(self.training_set) + len(self.test_set))]
        self.targets = [None for i in range(len(self.training_set) + len(self.test_set))]
        self.grouping = [None for i in range(len(self.training_set) + len(self.test_set))]
        if training_extra_tasks:
            self.extra_tasks = [(training_extra_tasks[t_id][0],
                                 [None for i in range(len(self.training_set) + len(self.test_set))])
                                for t_id in range(len(training_extra_tasks))]

    def add_train_test_paths(self, training_base_path: str, testing_base_path: str):
        self.training_set.base_path = training_base_path
        self.test_set.base_path = testing_base_path

    def k_folds(self, random_state=None, n_splits=5):
        """
        Produces a k_fold training/test split generator, depending on the task type

        :param n_splits: number of folds
        :param random_state: optional int for reproducability purposes
        :return: _BaseKFold object
        """
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
            kf = GroupKFold(n_splits=n_splits)
            return kf.split(inputs, groups=self.grouping)

        if self.task.classification_type == 'multi-label':
            kf = IterativeStratification(n_splits=n_splits)
            targets = np.array(targets)
            return kf.split(inputs, targets)
        else:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            targets = LabelEncoder().fit_transform([''.join(str(l)) for l in targets])
            return kf.split(inputs, targets)

    def get_split_by_index(self, train_index, test_index):
        """
        Returns the training and test set from the given lists of indexes

        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :return: the taskdataset for the training and test data
        """

        # if self.check_train_test_present():
        #     self.return_data()

        x_train = [self.inputs[i] for i in train_index]
        y_train = [self.targets[i] for i in train_index]
        grouping_train = [self.grouping[i] for i in train_index] if self.grouping else None
        extra_tasks_train = [(t[0], [t[1][i] for i in train_index]) for t in self.extra_tasks] if self.extra_tasks else None

        x_val = [self.inputs[i] for i in test_index]
        y_val = [self.targets[i] for i in test_index]
        grouping_val = [self.grouping[i] for i in test_index] if self.grouping else None
        extra_tasks_val = [(t[0], [t[1][i] for i in test_index]) for t in self.extra_tasks] if self.extra_tasks else None

        # for i in train_index:
        #     x_train.append(soft_pop(self.inputs, i))
        #     y_train.append(soft_pop(self.targets, i))
        #     if self.grouping:
        #         grouping_train.append(soft_pop(self.grouping, i))
        #
        #     extra_tasks_train = [(extra_tasks_train[t_id][0],
        #                           extra_tasks_train[t_id][1].append(soft_pop(self.extra_tasks[t_id][1], i)))
        #                          for t_id in range(len(extra_tasks_train))] if self.extra_tasks else None
        # for i in test_index:
        #     x_val.append(soft_pop(self.inputs, i))
        #     y_val.append(soft_pop(self.targets, i))
        #     if self.grouping:
        #         grouping_val.append(soft_pop(self.grouping, i))
        #
        #     extra_tasks_val = [(extra_tasks_val[t_id][0],
        #                         extra_tasks_val[t_id][1].append(soft_pop(self.extra_tasks[t_id][1], i)))
        #                        for t_id in range(len(extra_tasks_val))] if self.extra_tasks else None

        # if not self.extraction_method.scalers:
        #     self.extraction_method.scale_fit(x_train)

        self.training_set = TrainTaskDataset(inputs=x_train, targets=y_train,
                                             task=self.task, extraction_method=self.extraction_method,
                                             base_path=self.base_path, grouping=grouping_train,
                                             extra_tasks=extra_tasks_train, index_mode=self.index_mode)
        # self.training_indexes = train_index
        self.test_set = TestTaskDataset(inputs=x_val, targets=y_val,
                                        task=self.task,
                                        extraction_method=self.extraction_method,
                                        base_path=self.base_path, grouping=grouping_val,
                                        extra_tasks=extra_tasks_val, index_mode=self.index_mode)
        # self.test_indexes = test_index

    # def return_data(self):
    #     """
    #     returns the data in the training and test sets to the holding dataset in order to prevent memory waste
    #     :return:
    #     """
    #
    #     for i in range(len(self.training_set)):
    #         self.inputs[self.training_indexes[i]] = soft_pop(self.training_set.inputs, i)
    #         self.targets[self.training_indexes[i]] = soft_pop(self.training_set.targets, i)
    #         if self.grouping:
    #             self.grouping[self.training_indexes[i]] = soft_pop(self.training_set.grouping, i)
    #
    #         if self.extra_tasks:
    #             for t_id in range(len(self.extra_tasks)):
    #                 self.extra_tasks[t_id][1][self.training_indexes[i]] = soft_pop(self.training_set.extra_tasks[t_id][1], i)
    #     for i in range(len(self.test_set)):
    #         self.inputs[self.test_indexes[i]] = soft_pop(self.test_set.inputs, i)
    #         self.targets[self.test_indexes[i]] = soft_pop(self.test_set.targets, i)
    #         if self.grouping:
    #             self.grouping[self.test_indexes[i]] = soft_pop(self.test_set.grouping, i)
    #         if self.extra_tasks:
    #             for t_id in range(len(self.extra_tasks)):
    #                 self.extra_tasks[t_id][1][self.test_indexes[i]] = soft_pop(self.test_set.extra_tasks[t_id][1], i)

    def save_split_scalers(self, random_state, n_splits=5):
        """
        Calculates and saves the scaler objects for normalization for each fold, given a random_state, in separate files
        :param n_splits: number of folds
        :param random_state: optional int for reproducability purposes
        """

        i = 0
        for train, _ in self.k_folds(random_state, n_splits):
            x_train = [self.inputs[ind] for ind in range(len(self.inputs)) if ind in train]
            self.extraction_method.scale_fit(x_train)
            scalers = self.extraction_method.scalers
            path = os.path.join(self.base_path,
                                'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                                  random_state, i))
            joblib.dump(value=scalers, filename=path, protocol=pickle.HIGHEST_PROTOCOL)
            i += 1

    def load_split_scalers(self, fold, random_state):
        """
        Loads the scaler objects for dataset normalization purposes for the specified fold and random state
        :param fold: The fold to load
        :param random_state: The random state the five fold datasets was created with
        """
        path = os.path.join(self.base_path,
                            'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                              random_state, fold))
        self.extraction_method.scalers = joblib.load(path)

    def check_split_scalers(self, fold, random_state):
        path = os.path.join(self.base_path,
                            'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                              random_state, fold))
        return os.path.isfile(path)

    # def normalize_fit(self):
    #     self.training_set.normalize_fit()

    def prepare_inputs(self):
        if self.check_train_test_present():
            self.training_set.prepare_inputs()
            self.test_set.prepare_inputs()
        else:
            super().prepare_inputs()

    def normalize_inputs(self):
        self.training_set.normalize_inputs()
        self.test_set.normalize_inputs()


