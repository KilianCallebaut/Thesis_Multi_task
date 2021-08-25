import math
import os

import dcase_util
import joblib
from dcase_util.datasets import TUTSoundEvents_2017_DevelopmentSet

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiLabelTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class DCASE2017_SE(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SE_{}"
    wav_folder = 'F:\\Thesis_Datasets\\DCASE2017\\TUT-sound-events-2017-development\\audio\\'

    def __init__(self, **kwargs):
        print('start DCASE2017 SE')
        super().__init__(**kwargs)
        print('done DCASE2017 SE')

    def get_path(self):
        return os.path.join(self.get_base_path()['base_path'], 'DCASE2017_SE.obj')

    def get_base_path(self):
        return dict(base_path=self.object_path.format('train'))

    def check_files(self, extraction_method):
        return super().check_files(extraction_method) and \
               os.path.isfile(self.get_path())

    def read_files(self, taskDataset: HoldTaskDataset):
        info = joblib.load(self.get_path())
        self.audio_files = info['audio_files']
        super().read_files(taskDataset)

    def write_files(self, taskDataset):
        super().write_files(taskDataset)
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())

    def load_files(self):
        self.devdataset = TUTSoundEvents_2017_DevelopmentSet(
            data_path='F:\\Thesis_Datasets\\DCASE2017\\',
            log_system_progress=False,
            show_progress_in_console=True,
            use_ascii_progress_bar=True,
            name='TUTSoundEvents_2017_DevelopmentSet',
            fold_list=[1, 2, 3, 4],
            evaluation_mode='folds',
            storage_name='TUT-sound-events-2017-development'

        ).initialize()
        if os.path.isfile(self.get_path()):
            info = joblib.load(self.get_path())
            self.audio_files = info['audio_files']
            return

        self.audio_files = self.devdataset.audio_files

    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict):
        perc = 0
        for audio_idx in range(len(self.audio_files)):
            audio_cont = dcase_util.containers.AudioContainer().load(
                filename=self.audio_files[audio_idx]
            )
            segmented_signals, meta = audio_cont.segments(segment_length_seconds=1.0)
            for ss in segmented_signals:
                read_wav = self.preprocess_signal((ss, audio_cont.fs),
                                                  **preprocess_parameters)
                taskDataset.extract_and_add_input(read_wav)
            while perc < (audio_idx / len(self.audio_files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        print("Calculating input done")

    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
        distinct_labels = self.devdataset.scene_labels()
        targets = []

        for file_id in range(len(self.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            audio_cont = dcase_util.containers.AudioContainer().load(filename=self.audio_files[file_id])
            annotations = self.devdataset.meta.filter(self.audio_files[file_id])
            target = annotations.to_event_roll(label_list=self.devdataset.event_labels(),
                                               time_resolution=1,
                                               length_seconds=math.floor(audio_cont.duration_sec)).data
            for t in target.T:
                targets.append(t)
            print(file_id / len(self.audio_files))

        taskDataset.add_task_and_targets(
            targets=targets,
            task=MultiLabelTask(name='DCASE2017_SE', output_labels=distinct_labels)
        )
