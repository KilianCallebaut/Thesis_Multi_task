import os
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold, BaseCrossValidator
from sklearn.preprocessing import LabelEncoder
from skmultilearn.model_selection import IterativeStratification

from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset


class HoldTaskDataset(TaskDataset):
    """
    The TaskDataset object that unifies and splits the training/test sets
    """

    def __init__(self,
                 extraction_method: ExtractionMethod,
                 base_path: str = '',
                 index_mode=False,
                 testing_base_path: str = None):
        super().__init__(extraction_method, base_path, index_mode)
        assert (base_path,
                'A base_path should be given')

        self.test_set = TaskDataset(
            extraction_method=self.extraction_method,
            base_path=testing_base_path if testing_base_path else os.path.join(self.base_path, 'test_set'),
            index_mode=self.index_mode
        )

        self.test_indexes = []

    ########################################################################################################
    # Initialization
    ########################################################################################################

    # def add_test_input(self, sig_samplerate: Tuple[np.ndarray, int]):
    #     """
    #     Adds an input tensor to the testdataset
    #     :param sig_samplerate: The tuple with the signal object and the samplerate
    #     """
    #     self.test_indexes.append(len(self.test_indexes))
    #     self.test_set.add_input(sig_samplerate)
    #
    # def add_test_task_and_targets(self, task: Task, targets: List[List[int]]):
    #     """
    #     Adds a task object with a list of targets to the dataset.
    #     The targets should be in the same order as their corresponding inputs.
    #     :param task: The task object to add
    #     :param targets: The list of target vectors to add
    #     """
    #     assert task in self.get_all_tasks(), 'The task must already be in the dataset'
    #     self.test_set.add_task_and_targets(task, targets)
    #
    # def add_test_grouping(self, grouping: List[int]):
    #     """
    #     Adds the grouping list to the test set.
    #     The groupings should be in the same order as their corresponding inputs
    #     :param grouping: optional Grouping list for defining which data cannot be split up in k folds (see sklearn.model_selection.GroupKfold)
    #     """
    #     self.test_set.add_grouping(grouping=[g + max(self.grouping) for g in grouping])

    ########################################################################################################
    # Splitters
    ########################################################################################################
    def k_folds(self, random_state: int = None, n_splits: int = 5, kf: BaseCrossValidator = None):
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
        self.return_data()

        self.test_indexes = test_index

        self.distribute_data()
        # self.test_set.inputs = [self.inputs[i] for i in test_index]
        # self.test_set.targets = [self.targets[i] for i in test_index]
        # self.test_set.grouping = [self.grouping[i] for i in test_index] if self.grouping else None
        # self.test_set.extra_tasks = [(t[0], [t[1][i] for i in test_index]) for t in
        #                              self.extra_tasks] if self.extra_tasks else None
        #
        # self.inputs = [self.inputs[i] for i in train_index]
        # self.targets = [self.targets[i] for i in train_index]
        # self.grouping = [self.grouping[i] for i in train_index] if self.grouping else None
        # self.extra_tasks = [(t[0], [t[1][i] for i in train_index]) for t in
        #                     self.extra_tasks] if self.extra_tasks else None

    def return_data(self):
        """
        Returns the test set data to the main object
        """
        self.test_indexes.sort()
        for i in range(len(self.test_indexes)):
            self.inputs.insert(self.test_indexes[i], self.test_set.inputs[i])
            self.targets.insert(self.test_indexes[i], self.test_set.targets[i])
            self.grouping.insert(self.test_indexes[i], self.test_set.grouping[i])
            for t in self.extra_tasks:
                t[1].insert(self.test_indexes[i], self.test_set.extra_tasks[1][i])

    def distribute_data(self):
        """
        Distributes the data over the test set according to the saved test_indexes
        """
        self.test_indexes.sort(reverse=True)
        self.test_set.inputs = [self.inputs.pop(i) for i in self.test_indexes]
        self.test_set.targets = [self.targets.pop(i) for i in self.test_indexes]
        self.test_set.grouping = [self.grouping.pop(i) for i in self.test_indexes] if self.grouping else None
        self.test_set.extra_tasks = [(t[0], [t[1].pop(i) for i in self.test_indexes]) for t in
                                     self.extra_tasks] if self.extra_tasks else None

    def generate_train_test_set(self, random_state: int = None, n_splits=5, kf: BaseCrossValidator = None):
        """
        Generates and returns the train and test split
        :param random_state:
        :param n_splits:
        :param kf:
        :return:
        """
        if self.test_indexes:
            for i in range(n_splits):
                yield self, self.test_set
        else:
            for train, test in self.k_folds(random_state=random_state, n_splits=n_splits, kf=kf):
                self.get_split_by_index(train_index=train, test_index=test)
                yield self, self.test_set

    ########################################################################################################
    # Filtering
    ########################################################################################################
    def sample_labels(self, dic_of_labels_limits, random_state=None):
        """

        :param dic_of_labels_limits:
        :param random_state:
        :return:
        """
        self.return_data()
        super().sample_labels(dic_of_labels_limits=dic_of_labels_limits,
                              random_state=random_state)
        self.test_indexes = [i for i in self.test_indexes if self.inputs[i] in self.inputs]
        self.distribute_data()

    ########################################################################################################
    # Transformation
    ########################################################################################################
    def normalize_fit(self):
        assert len(self.test_set) > 0, 'scaling calculation should only happen on the training set'
        self.normalize_fit()

    def prepare_inputs(self):
        """
        Applies preparation of the input instances defined in the extraction method
        on each input instance in the TaskDataset.
        """
        super().prepare_inputs()
        if self.test_set:
            self.test_set.prepare_inputs()

    ########################################################################################################
    # I/O
    ########################################################################################################
    def save(self):
        super().save()
        if self.test_set:
            self.test_set.save()

    def load(self):
        super().load()
        if self.test_set:
            self.test_set.load()

    @staticmethod
    def check(extraction_method: ExtractionMethod, base_path: str = None,
              testing_base_path: str = None, index_mode: bool = False):
        """
        Checks if there are stored taskdatsets for this extraction method
        :param extraction_method: The extraction method object under which name the data is saved
        :param base_path: The path to the folder where the data is saved
        :param testing_base_path: The path to the folder where the predefined testing data is saved
        :param index_mode: Whether or not the data is stored in index_mode or not
        :return: Whether or not there is stored data
        """

        if base_path:
            TaskDataset.check(base_path=base_path,
                              extraction_method=extraction_method,
                              index_mode=index_mode)
        elif testing_base_path:
            return TaskDataset.check(base_path=base_path,
                                     extraction_method=extraction_method,
                                     index_mode=index_mode) \
                   and TaskDataset.check(base_path=testing_base_path,
                                         extraction_method=extraction_method,
                                         index_mode=index_mode)
        else:
            raise ValueError('Either a base_path or the training_base_path and testing_base_path should be given')
