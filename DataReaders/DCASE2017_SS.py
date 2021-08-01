import os
from typing import List

import joblib
import torch
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet, TUTAcousticScenes_2017_EvaluationSet
from numpy import long

from Tasks.Task import MultiClassTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

from DataReaders.DataReader import DataReader


class DCASE2017_SS(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SS_{}"
    # object_path = r"E:\Thesis_Results\Data_Readers\DCASE2017_SS_{}"
    wav_folder = 'F:\\Thesis_Datasets\\DCASE2017\\TUT-acoustic-scenes-2017-development\\audio\\'
    wav_folder_eval = 'F:\\Thesis_Datasets\\DCASE2017\\TUT-acoustic-scenes-2017-evaluation\\audio\\'

    # wav_folder = 'C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system-master' \
    #              '\\applications\\data\\TUT-acoustic-scenes-2017-development\\audio\\'
    # wav_folder_eval = 'C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system-master' \
    #                   '\\applications\\data\\TUT-acoustic-scenes-2017-evaluation\\audio\\'

    def __init__(self, extraction_method, **kwargs):
        print('start DCASE2017 SS')
        super().__init__(extraction_method, **kwargs)
        print('done DCASE2017 SS')

    def get_base_path(self):
        return dict(training_base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def get_path(self):
        return os.path.join(self.get_base_path()['training_base_path'],
                            'DCASE2017_SS.obj')

    def get_eval_path(self):
        return os.path.join(self.get_base_path()['testing_base_path'],
                            'DCASE2017_SS.obj')

    def check_files(self):
        return super().check_files() and \
               os.path.isfile(self.get_path()) and os.path.isfile(self.get_eval_path())

    def load_files(self):
        # MetaDataContainer(filename=)
        self.devdataset = TUTAcousticScenes_2017_DevelopmentSet(
            data_path='F:\\Thesis_Datasets\\DCASE2017\\',
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name='TUTAcousticScenes_2017_DevelopmentSet',
            fold_list=[1, 2, 3, 4],
            # fold_list=[1],
            evaluation_mode='folds',
            storage_name='TUT-acoustic-scenes-2017-development'

        ).initialize()
        self.evaldataset = TUTAcousticScenes_2017_EvaluationSet(
            data_path='F:\\Thesis_Datasets\\DCASE2017\\',
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name=r'TUTAcousticScenes_2017_EvaluationSet',
            fold_list=[1, 2, 3, 4],
            evaluation_mode='folds',
            storage_name='TUT-acoustic-scenes-2017-evaluation'
        ).initialize()

        self.audio_files = self.devdataset.audio_files
        self.audio_files_eval = self.evaldataset.audio_files

    def read_files(self) -> HoldTaskDataset:
        self.audio_files = joblib.load(self.get_path())
        self.audio_files_eval = joblib.load(self.get_eval_path())
        return super().read_files()

    def write_files(self, taskDataset: HoldTaskDataset):
        super().write_files(taskDataset=taskDataset)
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())

        dict = {'audio_files_eval': self.audio_files_eval}
        joblib.dump(dict, self.get_eval_path())

    def calculate_input(self, files, resample_to=None) -> List[torch.tensor]:
        return self.calculate_features(files, resample_to)

    def calculate_features(self, files, resample_to):
        inputs = []
        perc = 0
        for audio_idx in range(len(files)):
            read_wav = self.load_wav(files[audio_idx], resample_to)
            inputs.append(self.extraction_method.extract_features(read_wav))
            if perc < (audio_idx / len(files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        print("Calculating input done")
        return inputs

    def calculate_taskDataset(self, **kwargs) -> HoldTaskDataset:
        distinct_labels = self.devdataset.scene_labels()
        targets = []

        for file_id in range(len(self.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.devdataset.meta.filter(self.audio_files[file_id])[0]

            target = [long(distinct_labels[label_id] == annotations.scene_label) for label_id in
                      range(len(distinct_labels))]
            targets.append(target)

            print(file_id / len(self.audio_files))

        targets_val = []
        for file_id in range(len(self.audio_files_eval)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.evaldataset.meta.filter(self.audio_files_eval[file_id])[0]

            target = [long(distinct_labels[label_id] == annotations.scene_label) for label_id in
                      range(len(distinct_labels))]
            targets_val.append(target)
            print(file_id / len(self.audio_files_eval))

        inputs = self.calculate_input(files=self.audio_files, **kwargs)
        inputs_val = self.calculate_input(files=self.audio_files_eval, **kwargs)

        taskDataset = self.__create_taskDataset__()
        taskDataset.initialize_train_test(task=MultiClassTask(name='DCASE2017_SS', output_labels=distinct_labels),
                                          training_inputs=inputs,
                                          training_targets=targets,
                                          testing_inputs=inputs_val,
                                          testing_targets=targets_val)

        # taskDataset.prepare_inputs()
        return taskDataset
