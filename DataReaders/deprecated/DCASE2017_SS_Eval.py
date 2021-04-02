import multiprocessing
import os

import joblib
import librosa
from dcase_util.containers import MetaDataContainer
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet, TUTAcousticScenes_2017_EvaluationSet
from numpy import long

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset
import scipy.io.wavfile as wav


class DCASE2017_SS_Eval(DataReader):
    # audioset_path = r"E:\Thesis_Results\Data_Readers\DCASE2017_SS_Eval.p"
    audioset_path = r"E:\Thesis_Results\Data_Readers\DCASE2017_SS_Eval"
    base_path = 'C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system-master' \
                '\\applications\\data\\TUT-acoustic-scenes-2017-evaluation\\audio\\'

    def __init__(self, extraction_method, **kwargs):
        self.count = 0
        if 'object_path' in kwargs:
                  object_path = kwargs.pop('object_path')
        if self.check_files(extraction_method):
            self.read_files(extraction_method)
        else:
            self.load_files()
            self.calculate_taskDataset(extraction_method, **kwargs)
            self.write_files(extraction_method)

    def get_path(self):
        return os.path.join(self.get_base_path(), 'DCASE2017_SS_Eval.obj')

    def get_base_path(self):
        return self.audioset_path

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        # MetaDataContainer(filename=)

        self.evaldataset = TUTAcousticScenes_2017_EvaluationSet(
            data_path=r'C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system'
                      r'-master\\applications\\data\\',
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name=r'TUTAcousticScenes_2017_EvaluationSet',
            fold_list=[1, 2, 3, 4],
            # fold_list=[1],
            evaluation_mode='folds',
            storage_name='TUT-acoustic-scenes-2017-evaluation'
        ).initialize()
        self.audio_files = self.evaldataset.audio_files

    def read_files(self, extraction_method):
        info = joblib.load(self.get_path())
        self.audio_files = info['audio_files']
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)
        print('Reading SS Eval done')

    def write_files(self, extraction_method):
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method)

    def toTaskDataset(self):
        return self.taskDataset

    def calculate_taskDataset(self, method, **kwargs):
        distinct_labels = self.evaldataset.scene_labels()
        targets = []

        for file_id in range(len(self.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.evaldataset.meta.filter(self.audio_files[file_id])[0]

            target = [long(distinct_labels[label_id] == annotations.scene_label) for label_id in
                      range(len(distinct_labels))]
            targets.append(target)

            print(file_id / len(self.audio_files))

        inputs = self.calculate_input(method, **kwargs)
        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='DCASE2017_SS_eval',
                                       labels=distinct_labels,
                                       output_module='softmax')

    def calculate_input(self, method, **kwargs):
        inputs = []
        perc = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for audio_idx in range(len(self.audio_files)):
            read_wav = self.load_wav(self.audio_files[audio_idx], resample_to)
            inputs.append(self.extract_features(method, read_wav, **kwargs))
            if perc < (audio_idx / len(self.audio_files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return self.standardize_input(inputs)

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)

    def prepare_taskDatasets(self, test_size):
        pass

    def toTestTaskDataset(self):
        pass
