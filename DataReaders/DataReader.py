import math
from abc import abstractmethod, ABC
from typing import Tuple

import librosa
import librosa.display
import numpy as np
import soundfile
from scipy import signal

from DataReaders.ExtractionMethod import ExtractionMethod
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class DataReader(ABC):
    extractor = None

    def __init__(self,
                 object_path: str = None,
                 data_path: str = None,
                 **kwargs):
        if object_path:
            self.object_path = object_path
        if data_path:
            self.data_path = data_path
        self.skip_files=[]

    def return_taskDataset(self,
                           extraction_method: ExtractionMethod,
                           recalculate: bool = False,
                           index_mode: bool = False,
                           preprocess_parameters=None,
                           **kwargs) -> HoldTaskDataset:
        """
        Either reads or calculates the HoldTaskDataset object from the dataset
        :param index_mode: Get matrices by reading from the disk
        :param recalculate: Defines whether to automatically read the extracted data if available or not
        :param extraction_method: The extraction method object to extract the inputs with
        :param preprocess_parameters: The preprocessing parameters see preprocess_signal
        :return: HoldTaskDataset: The standardized object
        """

        if preprocess_parameters is None:
            preprocess_parameters = {}
        taskDataset = self.__create_taskDataset__(extraction_method, index_mode)
        if self.check_files(taskDataset, **kwargs) and not recalculate:
            print('reading')
            self.read_files(taskDataset, **kwargs)
        else:
            print('calculating')
            self.load_files()
            self.calculate_input(taskDataset=taskDataset,
                                 preprocess_parameters=preprocess_parameters,
                                 **kwargs)
            self.calculate_taskDataset(taskDataset, **kwargs)
            taskDataset.validate()
            self.write_files(taskDataset, **kwargs)
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
    def get_task_name(self) -> str:
        pass

    @abstractmethod
    def load_files(self):
        """
        Load in the required files from the dataset to later extract the input and targets from
        """
        pass

    @abstractmethod
    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict, **kwargs):
        """
        Extract and add the input tensors to the taskDataset object using add_input for complete tensors,
        extract_and_add_input for extracting and adding feature matrices using the extraction_method object.

        If a predefined test set needs to be added, utilize the same methods on the taskDataset.test_set
        to fill the dataset with inputs

        :param taskDataset: The standardized HoldTaskDataset object to fill
        :param preprocess_parameters:
        Preprocessing parameters if utilising the preprocess_signal method on the time series, see preprocess_signal
        """
        pass

    @abstractmethod
    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
        """
        Fill the rest of the TaskDataset inputs with targets, tasks and possible groupings.
        :param taskDataset: The standardized HoldTaskDataset object to fill
        :param kwargs: Any additional parameters that the function needs
        """
        pass

    def __create_taskDataset__(self, extraction_method: ExtractionMethod, index_mode) -> HoldTaskDataset:
        assert 'base_path' in self.get_base_path() or (
                'training_base_path' in self.get_base_path() and 'testing_base_path' in self.get_base_path()), 'base_path or training_base_path and testing_base_path keys required'
        return HoldTaskDataset(extraction_method=extraction_method,
                               index_mode=index_mode,
                               **self.get_base_path())

    def check_files(self, taskDataset, **kwargs):
        return taskDataset.check(self.get_task_name())

    def read_files(self, taskDataset: HoldTaskDataset, **kwargs):
        taskDataset.load(self.get_task_name())

    def write_files(self, taskDataset: HoldTaskDataset, **kwargs):
        taskDataset.save()

    def resample(self, sig, sample_rate, resample_to):
        secs = len(sig) / sample_rate
        return signal.resample(sig, int(secs * resample_to)), resample_to

    def preprocess_signal(self,
                          sig_samplerate: Tuple[np.ndarray, int],
                          resample_to: int = None,
                          mono: bool = False) -> Tuple[np.ndarray, int]:

        if sig_samplerate[0].dtype != np.dtype('float32'):
            sig_samplerate = (sig_samplerate[0].astype('float32'), sig_samplerate[1])

        if len(sig_samplerate[0].shape) > 1:
            channel = np.argmin(sig_samplerate[0].shape).item()
            if mono:
                sig_samplerate = (np.mean(sig_samplerate[0], axis=channel), sig_samplerate[1])

        if resample_to is not None:
            sig_samplerate = (librosa.core.resample(sig_samplerate[0], sig_samplerate[1], resample_to), resample_to)
        return sig_samplerate

    def time_split_signal(self,
                          sig_sample_rate: Tuple[np.ndarray, int],
                          time_resolution: float,
                          time_overlap: float = None):
        window_size = math.floor(sig_sample_rate[1] * time_resolution)
        window_hop = math.floor(sig_sample_rate[1] * time_overlap) if time_overlap else window_size

        windowed_signals = []
        start_frame = window_size

        end_frame = start_frame + window_hop * math.floor(
            (float(sig_sample_rate[0].shape[0] - start_frame) / window_hop))
        for frame_idx in range(start_frame, end_frame + 1, window_hop):
            window = sig_sample_rate[0][frame_idx - window_size:frame_idx, :]
            assert window.shape[0] == window_size
            windowed_signals.append(window)

        if start_frame > sig_sample_rate[0].shape[0]:
            window = np.vstack(
                [sig_sample_rate[0],
                 np.zeros(start_frame - sig_sample_rate[0].shape[0], *sig_sample_rate[0].shape[1:])])
            windowed_signals.append(window)
        elif end_frame < sig_sample_rate[0].shape[0]:
            window = sig_sample_rate[0][sig_sample_rate[0].shape[0] - window_size:sig_sample_rate[0].shape[0], :]
            windowed_signals.append(window)
        return windowed_signals

    def load_wav(self, loc) -> Tuple[np.ndarray, int]:
        sig, fs = soundfile.read(loc)

        if sig.shape[0] == 0:
            return None
        return sig, fs
