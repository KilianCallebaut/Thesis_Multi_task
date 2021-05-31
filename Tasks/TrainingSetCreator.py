import types
from typing import Optional

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.TaskDataset import TaskDataset


# get the prepare inputs in here iso the datareaders
# think how you will calculate, save and load the scalers

class ConcatTrainingSetCreator:
    def __init__(self, training_sets: list, test_sets: Optional[list], dics_of_labels_limits: list,
                 random_state: Optional[int]):
        self.training_creators = []
        self.random_state = random_state
        training_with_tests = [tst.task.name.split('_')[0] for tst in test_sets]
        for t_id in range(len(training_sets)):
            if not training_sets[t_id].task.name in training_with_tests:
                self.training_creators.append(TrainingSetCreator(dataset=training_sets[t_id],
                                                                 dic_of_labels_limits=dics_of_labels_limits[t_id],
                                                                 random_state=random_state,
                                                                 nr_runs=None,
                                                                 test_dataset=None))
            else:
                self.training_creators.append(TrainingSetCreator(dataset=training_sets[t_id],
                                                                 test_dataset=test_sets[training_with_tests.index(
                                                                     training_sets[t_id].task.name)],
                                                                 dic_of_labels_limits=dics_of_labels_limits[t_id],
                                                                 random_state=random_state,
                                                                 nr_runs=5 if len(training_with_tests) < len(
                                                                     training_sets) else 1))

    def generate_concats(self):
        gens = [tc.generate_train_test() for tc in self.training_creators]
        for i in range(self.training_creators[0].nr_runs):
            train_tests = [next(gen) for gen in gens]
            concat_training = ConcatTaskDataset([tt[0] for tt in train_tests])
            concat_tests = ConcatTaskDataset([tt[1] for tt in train_tests])
            yield concat_training, concat_tests

    def prepare_scalers(self):
        for tc in self.training_creators:
            tc.dataset.save_split_scalers(self.random_state)
            tc.dataset.save_scalers()

    def prepare_for_index_mode(self):
        for tc in self.training_creators:
            tc.dataset.inputs = tc.dataset.prepare_inputs()
            if tc.test_dataset:
                tc.test_dataset.inputs = tc.test_dataset.prepare_inputs(window_size=tc.dataset.inputs[0].shape[0])
            tc.dataset.to_index_mode()


class TrainingSetCreator:

    def __init__(self, dataset: TaskDataset, test_dataset: Optional[TaskDataset], dic_of_labels_limits: Optional[dict],
                 random_state: Optional[int], nr_runs: Optional[int]):
        self.dataset = dataset
        self.dic_of_labels_limits = dic_of_labels_limits
        self.random_state = random_state
        self.test_dataset = test_dataset

        if not self.dataset.index_mode:
            self.dataset.prepare_inputs()
        self.dataset.sample_labels(self.dic_of_labels_limits, random_state)

        if not test_dataset:
            self.nr_runs = 5
        else:
            if not isinstance(nr_runs, int):
                self.nr_runs = 1
            if not self.test_dataset.index_mode:
                self.test_dataset.prepare_inputs(window_size=self.dataset.inputs[0].shape[0])
                self.test_dataset.sample_labels(self.dic_of_labels_limits, random_state)

    def generate_train_test(self):
        if not self.test_dataset:
            return self.generate_five_fold()
        else:
            return self.return_train_val_set()

    def generate_five_fold(self):

        iterator = self.dataset.k_folds(random_state=self.random_state)
        for fold in range(self.nr_runs):
            try:
                print("fold: {}".format(fold))
                self.load_scaler(fold=fold)
                train_indices, test_indices = next(iterator)
                yield self.dataset.get_split_by_index(train_indices, test_indices, fold=fold,
                                                      random_state=self.random_state)
            except StopIteration:
                break

    def return_train_val_set(self):
        self.load_scaler()
        self.test_dataset.extraction_method = self.dataset.extraction_method
        for i in range(self.nr_runs):
            print("fold: {}".format(i))
            yield self.dataset, self.test_dataset

    def load_scaler(self, **kwargs):
        if 'fold' in kwargs:
            self.dataset.load_split_scalers(kwargs.get('fold'), self.random_state)
        else:
            self.dataset.load_scalers()
