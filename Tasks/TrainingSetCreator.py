from typing import Optional, List

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.TaskDataset import TaskDataset
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class ConcatTrainingSetCreator:
    def __init__(self, random_state: Optional[int], nr_runs: Optional[int] = 5):
        self.training_creators = []
        self.random_state = random_state
        self.nr_runs = nr_runs
        # for t_id in range(len(training_sets)):
        #     set = training_sets[t_id]
        #     if set.index_mode:
        #         if not set.has_index_mode():
        #             raise Exception('Index files must be created first')
        #         self.training_creators.append(IndexModeTrainingSetCreator(dataset=set,
        #                                                                   dic_of_labels_limits=dics_of_labels_limits[
        #                                                                       t_id],
        #                                                                   random_state=random_state,
        #                                                                   nr_runs=nr_runs))
        #     else:
        #         self.training_creators.append(TrainingSetCreator(dataset=set,
        #                                                          dic_of_labels_limits=dics_of_labels_limits[t_id],
        #                                                          random_state=random_state,
        #                                                          nr_runs=nr_runs))

    def add_dataset(self, dataset: HoldTaskDataset, dic_of_labels_limits: dict):
        self.training_creators.append(TrainingSetCreator(
            dataset=dataset,
            dic_of_labels_limits=dic_of_labels_limits,
            random_state=self.random_state,
            nr_runs=self.nr_runs
        ))

    def generate_concats(self):
        gens = [tc.generate_train_test() for tc in self.training_creators]
        for i in range(self.training_creators[0].nr_runs):
            train_tests = [next(gen) for gen in gens]
            concat_training = ConcatTaskDataset([tt[0] for tt in train_tests])
            concat_tests = ConcatTaskDataset([tt[1] for tt in train_tests])
            yield concat_training, concat_tests


class TrainingSetCreator:

    def __init__(self, dataset: HoldTaskDataset, dic_of_labels_limits: Optional[dict],
                 random_state: Optional[int], nr_runs: Optional[int]):
        self.dataset = dataset
        self.dic_of_labels_limits = dic_of_labels_limits
        self.random_state = random_state
        self.nr_runs = nr_runs
        self.dataset.sample_labels(self.dic_of_labels_limits, self.random_state)

        # self.dataset.prepare_inputs(**kwargs)

        if dataset.check_train_test_present():
            if not isinstance(nr_runs, int):
                self.nr_runs = 1
        elif not isinstance(nr_runs, int):
            self.nr_runs = 5

    def generate_train_test(self, **kwargs):
        if self.dataset.check_train_test_present():
            return self.return_train_val_set()
        else:
            return self.generate_k_fold()

    def generate_k_fold(self):
        iterator = self.dataset.k_folds(random_state=self.random_state, n_splits=self.nr_runs)
        for fold in range(self.nr_runs):
            try:
                print("fold: {}".format(fold))

                train_indices, test_indices = next(iterator)
                self.dataset.get_split_by_index(train_indices, test_indices)
                self.dataset.training_set.normalize_fit()

                yield self.dataset.training_set, self.dataset.test_set
            except StopIteration:
                break

    def return_train_val_set(self):
        self.dataset.training_set.normalize_fit()
        for i in range(self.nr_runs):
            print("fold: {}".format(i))
            yield self.dataset.training_set, self.dataset.test_set
