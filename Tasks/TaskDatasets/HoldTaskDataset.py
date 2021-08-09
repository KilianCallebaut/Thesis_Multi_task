import os
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold
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
                 extraction_method: ExtractionMethod,
                 base_path: str = '',
                 index_mode=False,
                 training_base_path: str = None,
                 testing_base_path: str = None):
        super().__init__(extraction_method, base_path, index_mode)
        assert (base_path or (training_base_path and testing_base_path),
                'Either a base_path or a training_base_path and testing_base_path should be given')
        self.training_set = TrainTaskDataset(
            extraction_method=self.extraction_method,
            base_path=training_base_path if training_base_path else os.path.join(self.base_path, 'train_set'),
            index_mode=self.index_mode
        )
        self.test_set = TestTaskDataset(
            extraction_method=self.extraction_method,
            base_path=testing_base_path if testing_base_path else os.path.join(self.base_path, 'test_set'),
            index_mode=self.index_mode
        )


    def check_train_test_present(self):
        """
        Checks if this HoldTaskDataset has predefined train/test split
        :return: Bool
        """
        return len(self.training_set.inputs) > 0 and len(self.test_set.inputs) > 0

    ########################################################################################################
    # Setters
    ########################################################################################################

    def initialize_train_test(self, task: Task, training_inputs: List[torch.tensor], training_targets: List[List],
                              testing_inputs: List[torch.tensor], testing_targets: List[List],
                              training_grouping: List[int] = None, training_extra_tasks: List[Tuple[Task, List]] = None,
                              testing_grouping: List[int] = None, testing_extra_tasks: List[Tuple[Task, List]] = None):
        """
        Inserts the data for a dataset with predefined train/test splits
        :param task: The task object for the dataset
        :param training_inputs:  The list of input tensors for training
        :param training_targets: The list of targets for training
        :param testing_inputs: The list of input tensors for testing
        :param testing_targets: The list of targets for testing
        :param training_grouping: The grouping list defining which instances belong together for training
        :param training_extra_tasks: The list of extra task/target tuples for training
        :param testing_grouping: The grouping list defining which instances belong together for testing
        :param testing_extra_tasks: The list of extra task/target tuples for testing
        :return:
        """
        self.training_set.initialize(inputs=training_inputs, targets=training_targets, task=task,
                                     grouping=training_grouping, extra_tasks=training_extra_tasks)
        self.test_set.initialize(inputs=testing_inputs, targets=testing_targets, task=task,
                                 grouping=testing_grouping,
                                 extra_tasks=testing_extra_tasks)
        self.initialize(inputs=[],
                        targets=[],
                        task=task,
                        grouping=[],
                        extra_tasks=[])

    ########################################################################################################
    # Splitters
    ########################################################################################################
    def k_folds(self, random_state: int =None, n_splits: int =5, kf: _BaseKFold = None):
        """
        Produces a k_fold training/test split generator, depending on the task type

        :param kf: optional input for inserting own sklearn kfold splitter
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

        if kf:
            return kf.split(inputs, np.array(targets), groups=self.grouping)

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
        extra_tasks_train = [(t[0], [t[1][i] for i in train_index]) for t in
                             self.extra_tasks] if self.extra_tasks else None

        x_val = [self.inputs[i] for i in test_index]
        y_val = [self.targets[i] for i in test_index]
        grouping_val = [self.grouping[i] for i in test_index] if self.grouping else None
        extra_tasks_val = [(t[0], [t[1][i] for i in test_index]) for t in
                           self.extra_tasks] if self.extra_tasks else None

        self.training_set.initialize(inputs=x_train, targets=y_train,
                                     task=self.task, grouping=grouping_train,
                                     extra_tasks=extra_tasks_train)
        self.test_set.initialize(inputs=x_val, targets=y_val,
                                 task=self.task, grouping=grouping_val,
                                 extra_tasks=extra_tasks_val)

    def generate_train_test_set(self, random_state: int = None, n_splits=5, kf: _BaseKFold = None):
        """
        Generates and returns the train and test split
        :param random_state:
        :param n_splits:
        :param kf:
        :return:
        """
        if self.check_train_test_present():
            for i in range(n_splits):
                yield self.training_set, self.test_set
        else:
            for train, test in self.k_folds(random_state=random_state, n_splits=n_splits, kf=kf):
                self.get_split_by_index(train_index=train, test_index=test)
                yield self.training_set, self.test_set

    ########################################################################################################
    # Filtering
    ########################################################################################################

    def sample_labels(self, dic_of_labels_limits, random_state=None):
        if self.check_train_test_present():
            self.training_set.sample_labels(dic_of_labels_limits=dic_of_labels_limits, random_state=random_state)
            self.test_set.sample_labels(dic_of_labels_limits=dic_of_labels_limits, random_state=random_state)
        else:
            super().sample_labels(dic_of_labels_limits=dic_of_labels_limits, random_state=random_state)

    ########################################################################################################
    # Transformation
    ########################################################################################################
    def normalize_fit(self):
        self.training_set.normalize_fit()

    def prepare_inputs(self):
        if self.check_train_test_present():
            self.training_set.prepare_inputs()
            self.test_set.prepare_inputs()
        else:
            super().prepare_inputs()

    ########################################################################################################
    # I/O
    ########################################################################################################
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
        else:
            super().load()

    @staticmethod
    def check(extraction_method: ExtractionMethod, base_path: str = None,
              training_base_path: str = None, testing_base_path: str = None):
        if base_path:
            return super().check(base_path=base_path, extraction_method=extraction_method)
        elif training_base_path and testing_base_path:
            return super().check(base_path=training_base_path, extraction_method=extraction_method) and super().check(
                base_path=testing_base_path, extraction_method=extraction_method)
        else:
            raise ValueError('Either a base_path or the training_base_path and testing_base_path should be given')
