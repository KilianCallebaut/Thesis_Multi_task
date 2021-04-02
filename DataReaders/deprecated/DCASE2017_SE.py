import os

import librosa
from dcase_util.containers import MetaDataContainer
from dcase_util.datasets import TUTSoundEvents_2017_DevelopmentSet
from numpy import long

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset
import scipy.io.wavfile as wav

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


class DCASE2017_SE(DataReader):
    audioset_path = r"E:\Thesis_Results\Data_Readers\DCASE2017_SE.p"

    def __init__(self, extraction_method, **kwargs):
        if os.path.isfile(self.audioset_path):
            self.read_files()
        else:
            self.load_files()
            self.calculate_taskDataset(extraction_method, **kwargs)
            self.write_files()

    def load_files(self):
        self.devdataset = TUTSoundEvents_2017_DevelopmentSet(
            data_path='C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system-master\\applications\\data\\',
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name='TUTSoundEvents_2017_DevelopmentSet',
            fold_list=[1, 2, 3, 4],
            evaluation_mode='folds',
            storage_name=r'TUT-sound-events-2017-development'
        ).initialize()

    def read_files(self):
        info = cPickle.load(open(self.audioset_path, 'rb'))
        self.taskDataset = info['taskDataset']
        self.devdataset = info['devdataset']

    def write_files(self):
        dict = {'taskDataset': self.taskDataset,
                'devdataset': self.devdataset}
        cPickle.dump(dict, open(self.audioset_path, 'wb'))

    def toTaskDataset(self):
        return self.taskDataset

    def calculate_input(self, method, **kwargs):
        inputs = []
        perc = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for audio_idx in range(len(self.devdataset.audio_files)):
            read_wav = self.load_wav(self.devdataset.audio_files[audio_idx], resample_to)
            inputs.append(self.extract_features(method, read_wav, **kwargs))
            if perc < (audio_idx / len(self.devdataset.audio_files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return self.standardize_input(inputs)

    def calculate_taskDataset(self, method, **kwargs):
        distinct_labels = self.devdataset.scene_labels()
        targets = []

        for file_id in range(len(self.devdataset.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.devdataset.meta.filter(self.devdataset.audio_files[file_id])

            if 'time_resolution' in kwargs:
                target = annotations.to_event_roll(label_list=self.devdataset.event_labels(),
                                                   time_resolution=kwargs.get('time_resolution'))
            else:
                target = annotations.to_event_roll(label_list=self.devdataset.event_labels(),
                                                   time_resolution=0.2)

            target = [long(distinct_labels[label_id] == annotations.scene_label) for label_id in
                      range(len(distinct_labels))]
            targets.append(target)

            print(file_id / len(self.devdataset.audio_files))

        inputs = self.calculate_input(method, **kwargs)
        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='DCASE2017_SS',
                                       labels=distinct_labels,
                                       output_module='softmax')

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)

    # def calculate_taskDataset(self):
    #     pass
    #
    # def read_files(self):
    #     pass
    #
    # def write_files(self):
    #     pass


    #
    # def toTaskDataset(self):
    #     if self.taskDataset is None:
    #         inputs = []
    #         for file_id in range(len(self.devdataset.audio_files)):
    #             read_wav = librosa.load(self.devdataset.audio_files[file_id])
    #             self.samplerate_sig_list.append(read_wav)
    #             input_summary = self.extract_features(method, read_wav, **kwargs)
    #             inputs.append(input_summary)
    #             #targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
    #             meta = self.devdataset.meta.filter(self.devdataset.audio_files[file_id])
    #             target = meta.to_event_roll(label_list=self.devdataset.event_labels())
    #             print('ay')
    #
    #     #     targets = [for self.devdataset.]
    #     #     read_wav = wav.read(self.devdataset)
    #     #     self.taskDataset = TaskDataset(inputs=inputs,
    #     #                                    targets=,
    #     #                                    name=,
    #     #                                    labels=self.devdataset.meta_container['unique_event_labels'],
    #     #                                    output_module=,)
    #     # return self.taskDataset
    #     pass
