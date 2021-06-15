import random
from abc import abstractmethod, ABC

import librosa
import numpy as np
import soundfile
import torch
import matplotlib.pyplot as plt
import librosa.display
from scipy import signal
from sklearn.model_selection import KFold

from DataReaders.ExtractionMethod import extract_options
from Tasks.TaskDataset import TaskDataset


class DataReader(ABC):
    extractor = None

    def __init__(self, extraction_method, **kwargs):
        if 'object_path' in kwargs:
            self.object_path = kwargs.pop('object_path')
        if 'index_mode' in kwargs:
            self.index_mode = kwargs.pop('index_mode')
        else:
            self.index_mode = False

        self.extraction_method = extract_options[extraction_method]
        self.taskDataset = TaskDataset([], [], '', [], self.extraction_method, base_path=self.get_base_path(),
                                       index_mode=self.index_mode)

        if self.check_files(extraction_method):
            print('reading')
            self.read_files()
        else:
            print('calculating')
            self.load_files()
            self.calculate_taskDataset(**kwargs)
            self.write_files()


    @abstractmethod
    def get_path(self):
        pass

    @abstractmethod
    def get_base_path(self):
        pass

    @abstractmethod
    def check_files(self, extraction_method):
        pass

    @abstractmethod
    def load_files(self):
        pass

    @abstractmethod
    def read_files(self):
        pass

    @abstractmethod
    def write_files(self):
        pass

    @abstractmethod
    def calculate_input(self, resample_to=None, **kwargs):
        pass

    @abstractmethod
    def calculate_taskDataset(self, **kwargs):
        pass

    def sample_labels(self, taskDataset, dic_of_labels_limits):
        sampled_targets = taskDataset.targets
        sampled_inputs = taskDataset.inputs

        for l in dic_of_labels_limits.keys():
            label_set = [i for i in range(len(sampled_targets))
                         if sampled_targets[i][taskDataset.task.output_labels.index(l)] == 1]
            if len(label_set) > dic_of_labels_limits[l]:
                random_label_set = random.sample(label_set, dic_of_labels_limits[l])
                sampled_targets = [sampled_targets[i] for i in range(len(sampled_targets)) if
                                   (i not in label_set or i in random_label_set)]
                sampled_inputs = [sampled_inputs[i] for i in range(len(sampled_inputs)) if
                                  (i not in label_set or i in random_label_set)]
        return sampled_inputs, sampled_targets

    @abstractmethod
    def prepare_taskDatasets(self, test_size, dic_of_labels_limits, **kwargs):
        pass

    @abstractmethod
    def toTrainTaskDataset(self):
        pass

    @abstractmethod
    def toTestTaskDataset(self):
        pass

    @abstractmethod
    def toValidTaskDataset(self):
        pass

    def resample(self, sig, sample_rate, resample_to):
        secs = len(sig) / sample_rate
        return signal.resample(sig, int(secs * resample_to)), resample_to

    # def load_wav(self, loc, resample_to=None):
    #     fs, sig = wav.read(open(loc, 'rb'))
    #     if resample_to is not None:
    #         return self.resample(sig, fs, resample_to)
    #     return sig, fs

    def load_wav(self, loc, resample_to=None):
        sig, fs = soundfile.read(loc)
        if len(sig.shape) > 1:
            sig = np.mean(sig, axis=1)
        if sig.shape[0] == 0:
            return None
        if resample_to is not None and resample_to != fs:
            librosa.core.resample(sig, fs, resample_to)

        return sig, fs



