from abc import abstractmethod, ABC
from typing import List

import librosa
import librosa.display
import numpy as np
import soundfile
import torch
from scipy import signal

from DataReaders.ExtractionMethod import extract_options, ExtractionMethod
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class DataReader(ABC):
    extractor = None

    def __init__(self, extraction_method: ExtractionMethod, preparation_params: dict = None,
                 extraction_params: dict = None, object_path: str = None,
                 index_mode: bool = False, **kwargs):
        if object_path:
            self.object_path = object_path

        self.index_mode = index_mode

        if isinstance(extraction_method, str):
            self.extraction_method = extract_options[extraction_method](preparation_params, extraction_params)
        else:
            self.extraction_method = extraction_method

    def return_taskDataset(self) -> HoldTaskDataset:
        """
        Either reads or calculates the HoldTaskDataset object from the dataset
        :return: HoldTaskDataset: The standardized object
        """
        if self.check_files():
            print('reading')
            taskDataset = self.read_files()
        else:
            print('calculating')
            self.load_files()
            taskDataset = self.calculate_taskDataset(**kwargs)
            self.write_files(taskDataset)
        return taskDataset

    @abstractmethod
    def get_base_path(self) -> dict:
        """"
        Returns a dictionary with the path to the folder containing the extracted files.
        Has to have training_base_path and testing_base_path as keys if the dataset has predefined train/test sets,
        base_path as a key if not.
        See HoldTaskDataset
        """
        pass

    @abstractmethod
    def load_files(self):
        pass

    @abstractmethod
    def calculate_input(self, files, resample_to=None) -> List[torch.tensor]:
        pass

    @abstractmethod
    def calculate_taskDataset(self, **kwargs) -> HoldTaskDataset:
        pass

    def __create_taskDataset__(self) -> HoldTaskDataset:
        assert 'base_path' in self.get_base_path() or (
                'training_base_path' in self.get_base_path() and 'testing_base_path' in self.get_base_path()), 'base_path or training_base_path and testing_base_path keys required'
        return HoldTaskDataset(extraction_method=self.extraction_method,
                               index_mode=self.index_mode,
                               **self.get_base_path())

    def check_files(self):
        HoldTaskDataset.check(extraction_method=self.extraction_method, **self.get_base_path())

    def read_files(self) -> HoldTaskDataset:
        taskDataset = self.__create_taskDataset__()
        taskDataset.load()
        return taskDataset

    def write_files(self, taskDataset: HoldTaskDataset):
        taskDataset.save()

    def resample(self, sig, sample_rate, resample_to):
        secs = len(sig) / sample_rate
        return signal.resample(sig, int(secs * resample_to)), resample_to

    def load_wav(self, loc, resample_to=None):
        sig, fs = soundfile.read(loc)
        if len(sig.shape) > 1:
            sig = np.mean(sig, axis=1)
        if sig.shape[0] == 0:
            return None
        if resample_to is not None and resample_to != fs:
            librosa.core.resample(sig, fs, resample_to)

        return sig, fs
