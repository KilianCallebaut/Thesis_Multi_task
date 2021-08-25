import copy
from typing import Optional, List

from sklearn.model_selection import BaseCrossValidator

from DataReaders.DataReader import DataReader
from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.ConcatTaskDataset import ConcatTaskDataset


class ConcatTrainingSetCreator:

    def __init__(self,
                 random_state: Optional[int],
                 nr_runs: Optional[int] = 5):
        self.random_state = random_state
        self.nr_runs = nr_runs

        self.data_readers = {}
        self.extraction_methods = {}
        self.sample_rates = {}
        self.preproccesing = {}
        self.dics_of_label_limits = {}
        self.validators = {}
        self.taskdatasets = {}

        self.model = None

    def get_keys(self):
        return self.data_readers.keys()

    def reset_taskDatasets(self, class_list: List[str] = None):
        if class_list and self.taskdatasets:
            for k in self.taskdatasets.keys():
                if k not in class_list:
                    self.taskdatasets.pop(k)
        else:
            self.taskdatasets = {}

    def add_data_reader(self,
                        data_reader: DataReader):
        self.data_readers[type(data_reader).__name__] = data_reader

    def __add__pipe__(self,
                      addition,
                      dictionary: dict,
                      key: str = None):
        if not key:
            for k in self.data_readers.keys():
                dictionary[k] = copy.copy(addition)
            self.reset_taskDatasets()
        else:
            assert key in self.data_readers, 'There is not dataset with this key'
            dictionary[key] = addition
            self.reset_taskDatasets([k for k in self.data_readers.keys() if k is not key])

    def __get__pipe__(self,
                      dictionary: dict,
                      key: str = None):
        if key in dictionary:
            return dictionary[key]
        else:
            return None

    ###############
    # Extraction
    ###############

    def add_extraction_method(self,
                              extraction_method: ExtractionMethod,
                              key: str = None):
        self.__add__pipe__(addition=extraction_method,
                           dictionary=self.extraction_methods,
                           key=key)

    ###############
    # Signal Preprocessing
    ###############

    def add_sample_rate(self,
                        sample_rate: int,
                        key: str = None):
        self.__add__pipe__(addition=sample_rate,
                           dictionary=self.sample_rates,
                           key=key)

    def add_signal_preprocessing(self,
                                 preprocess_dict: dict,
                                 key: str = None):
        self.__add__pipe__(addition=preprocess_dict,
                           dictionary=self.preproccesing,
                           key=key)

    ###############
    # Filtering
    ###############

    def add_sampling(self,
                     dic_of_labels_limits: dict,
                     key: str = None):
        self.__add__pipe__(addition=dic_of_labels_limits,
                           dictionary=self.dics_of_label_limits,
                           key=key)

    ###############
    # Splitting
    ###############

    def add_cross_validator(self,
                            validator: BaseCrossValidator,
                            key: str = None):
        self.__add__pipe__(addition=validator,
                           dictionary=self.validators,
                           key=key)

    def create_taskdatasets(self, class_list: List[str] = None):
        self.reset_taskDatasets(class_list)
        for dr in self.data_readers.keys():
            if (class_list and dr not in class_list) or dr in self.taskdatasets.keys():
                continue
            tsk = self.data_readers[dr].return_taskDataset(
                extraction_method=self.__get__pipe__(key=dr, dictionary=self.extraction_methods),
                resample_to=self.__get__pipe__(key=dr,
                                               dictionary=self.sample_rates),
                **self.__get__pipe__(key=dr,
                                     dictionary=self.preproccesing)

            )
            if dr in self.dics_of_label_limits:
                tsk.sample_labels(self.__get__pipe__(key=dr, dictionary=self.dics_of_label_limits[dr]))

            tsk.prepare_fit()
            tsk.prepare_inputs()
            self.taskdatasets[dr] = tsk
        return self.taskdatasets

    def generate_training_splits(self, class_list: List[str] = None):
        self.create_taskdatasets(class_list)
        task_gens = []
        for dr in self.taskdatasets.keys():
            tsk = self.taskdatasets[dr]
            task_gens.append(
                tsk.generate_train_test_set(random_state=self.random_state,
                                            n_splits=self.nr_runs,
                                            kf=self.__get__pipe__(dictionary=self.validators,
                                                                  key=dr
                                                                  )
                                            )
            )

        fold = 0
        for train_tests in zip(*task_gens):
            print("Fold: {}".format(fold))
            fold += 1
            for t in train_tests:
                t[0].normalize_fit()
            yield ConcatTaskDataset([t[0] for t in train_tests]), ConcatTaskDataset([t[1] for t in train_tests]), fold
