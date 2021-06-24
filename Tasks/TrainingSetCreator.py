from typing import Optional, List

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.TaskDataset import TaskDataset
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class ConcatTrainingSetCreator:
    def __init__(self, training_sets: List[HoldTaskDataset], dics_of_labels_limits: list,
                 random_state: Optional[int], nr_runs: Optional[int] = 5,
                 prepare_args: Optional[dict] = dict()):
        self.training_creators = []
        self.random_state = random_state
        self.prepare_args = prepare_args
        for t_id in range(len(training_sets)):
            set = training_sets[t_id]
            if set.index_mode:
                if not set.has_index_mode():
                    raise Exception('Index files must be created first')
                self.training_creators.append(IndexModeTrainingSetCreator(dataset=set,
                                                                          dic_of_labels_limits=dics_of_labels_limits[
                                                                              t_id],
                                                                          random_state=random_state,
                                                                          nr_runs=nr_runs))
            else:
                self.training_creators.append(TrainingSetCreator(dataset=set,
                                                                 dic_of_labels_limits=dics_of_labels_limits[t_id],
                                                                 random_state=random_state,
                                                                 nr_runs=nr_runs,
                                                                 **prepare_args))

        # training_with_tests = [tst.task.name.split('_')[0] for tst in test_sets]
        # for t_id in range(len(training_sets)):
        #     if not training_sets[t_id].task.name.split('_')[0] in training_with_tests:
        #         self.training_creators.append(TrainingSetCreator(dataset=training_sets[t_id],
        #                                                          dic_of_labels_limits=dics_of_labels_limits[t_id],
        #                                                          random_state=random_state,
        #                                                          nr_runs=None,
        #                                                          test_dataset=None))
        #     else:
        #         self.training_creators.append(TrainingSetCreator(dataset=training_sets[t_id],
        #                                                          test_dataset=test_sets[training_with_tests.index(
        #                                                              training_sets[t_id].task.name.split('_')[0])],
        #                                                          dic_of_labels_limits=dics_of_labels_limits[t_id],
        #                                                          random_state=random_state,
        #                                                          nr_runs=5 if len(training_with_tests) < len(
        #                                                              training_sets) else 1))

    def generate_concats(self):
        gens = [tc.generate_train_test() for tc in self.training_creators]
        for i in range(self.training_creators[0].nr_runs):
            train_tests = [next(gen) for gen in gens]
            concat_training = ConcatTaskDataset([tt[0] for tt in train_tests])
            concat_tests = ConcatTaskDataset([tt[1] for tt in train_tests])
            yield concat_training, concat_tests

    def prepare_for_index_mode(self):

        for tc in self.training_creators:
            tc.dataset.prepare_inputs(**self.prepare_args)

            if tc.dataset.check_train_test_present():
                tc.dataset.training_set.save_scalers()
            else:
                tc.dataset.save_split_scalers(self.random_state)

            if tc.dataset.check_train_test_present():
                if not tc.dataset.training_set.has_index_mode():
                    tc.dataset.training_set.write_index_files()
                if not tc.dataset.test_set.has_index_mode():
                    tc.dataset.test_set.write_index_files()
            else:
                if not tc.dataset.has_index_mode():
                    tc.dataset.write_index_files()


class TrainingSetCreator:

    def __init__(self, dataset: HoldTaskDataset, dic_of_labels_limits: Optional[dict],
                 random_state: Optional[int], nr_runs: Optional[int],
                 **kwargs):
        self.dataset = dataset
        self.dic_of_labels_limits = dic_of_labels_limits
        self.random_state = random_state
        self.nr_runs = nr_runs

        self.dataset.prepare_inputs(**kwargs)

        if dataset.check_train_test_present():
            if not isinstance(nr_runs, int):
                self.nr_runs = 1
            # if not self.dataset.index_mode:
            #     self.test_dataset.prepare_inputs(window_size=self.dataset.inputs[0].shape[0])
        elif not isinstance(nr_runs, int):
            self.nr_runs = 5

    def generate_train_test(self, **kwargs):
        if self.dataset.check_train_test_present():
            return self.return_train_val_set()
        else:
            return self.generate_k_fold()

        # self.dataset.sample_labels(self.dic_of_labels_limits, self.random_state)
        # if self.test_dataset:
        #     return self.return_train_val_set()
        # else:
        #     return self.generate_five_fold()

    def generate_k_fold(self):
        self.dataset.sample_labels(self.dic_of_labels_limits, self.random_state)
        iterator = self.dataset.k_folds(random_state=self.random_state, n_splits=self.nr_runs)
        for fold in range(self.nr_runs):
            try:
                print("fold: {}".format(fold))

                train_indices, test_indices = next(iterator)
                self.dataset.get_split_by_index(train_indices, test_indices)
                self.dataset.normalize_fit()
                self.dataset.normalize_inputs()
                yield self.dataset.training_set, self.dataset.test_set
            except StopIteration:
                break

    def return_train_val_set(self):
        self.dataset.training_set.sample_labels(self.dic_of_labels_limits, self.random_state)
        self.dataset.prepare_inputs()
        self.dataset.normalize_fit()
        self.dataset.normalize_inputs()
        for i in range(self.nr_runs):
            print("fold: {}".format(i))
            yield self.dataset.training_set, self.dataset.test_set


class IndexModeTrainingSetCreator(TrainingSetCreator):

    def generate_train_test(self):
        if self.dataset.check_train_test_present():
            return self.return_train_val_set()
        else:
            return self.generate_k_fold()

    def generate_k_fold(self):
        self.dataset.sample_labels(self.dic_of_labels_limits, self.random_state)
        iterator = self.dataset.k_folds(random_state=self.random_state, n_splits=self.nr_runs)
        for fold in range(self.nr_runs):
            try:
                print("fold: {}".format(fold))
                train_indices, test_indices = next(iterator)
                self.dataset.get_split_by_index(train_indices, test_indices)
                self.load_scaler(fold)
                yield self.dataset.training_set, self.dataset.test_set
            except StopIteration:
                break

    def return_train_val_set(self):
        self.dataset.training_set.sample_labels(self.dic_of_labels_limits, self.random_state)
        self.load_scaler()
        for i in range(self.nr_runs):
            print("fold: {}".format(i))
            yield self.dataset.training_set, self.dataset.test_set

    def load_scaler(self, fold=None):
        if isinstance(fold, int) and self.dataset.check_split_scalers(fold, self.random_state):
            self.dataset.load_split_scalers(fold, self.random_state)
        elif self.dataset.check_scalers():
            self.dataset.training_set.load_scalers()
            self.dataset.test_set.extraction_method = self.dataset.training_set.extraction_method
        else:
            raise Exception('Scalers must be calculated and saved beforehand when using index mode')
