import os

import joblib
from dcase_util.datasets import TUTSoundEvents_2017_DevelopmentSet
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from DataReaders.ExtractionMethod import extract_options
from Tasks.TaskDataset import TaskDataset


class DCASE2017_SE(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SE_{}"
    wav_folder = 'F:\\Thesis_Datasets\\DCASE2017\\TUT-sound-events-2017-development\\audio\\'

    def __init__(self, extraction_method, **kwargs):
        print('start DCASE2017 SE')
        super().__init__(extraction_method, **kwargs)
        print('done DCASE2017 SE')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'DCASE2017_SE.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

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

        self.audio_files = self.devdataset.audio_files

    def read_files(self):
        info = joblib.load(self.get_path())
        self.audio_files = info['audio_files']
        self.taskDataset = TaskDataset([], [], '', [], self.extraction_method, base_path=self.get_base_path(),
                                       index_mode=self.index_mode)
        self.taskDataset.load(self.get_base_path())
        print('Reading SS done')

    def write_files(self):
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path())

    def calculate_input(self,  **kwargs):
        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        files = self.audio_files
        inputs = self.calculate_features(files, resample_to, **kwargs)
        print("Calculating input done")
        return inputs

    def calculate_features(self, files, resample_to, **kwargs):
        inputs = []
        perc = 0
        for audio_idx in range(len(files)):
            read_wav = self.load_wav(files[audio_idx], resample_to)
            inputs.append(self.extraction_method.extract_features(read_wav, **kwargs))
            if perc < (audio_idx / len(files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        print("Calculating input done")
        return inputs

    def calculate_taskDataset(self, **kwargs):
        distinct_labels = self.devdataset.scene_labels()
        targets = []

        for file_id in range(len(self.audio_files)):
            # targets in the form list Tensor 2d (nr_frames, nr_labels) of length nr_files
            annotations = self.devdataset.meta.filter(self.audio_files[file_id])
            target = annotations.to_event_roll(label_list=self.devdataset.event_labels(),
                                               time_resolution=1)

            targets.append(target)

            print(file_id / len(self.audio_files))

        inputs = self.calculate_input(**kwargs)

        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='DCASE2017_SE',
                                       labels=distinct_labels,
                                       extraction_method=self.extraction_method,
                                       base_path=self.get_base_path(),
                                       output_module='softmax',
                                       index_mode=self.index_mode)

    def prepare_taskDatasets(self, test_size, dic_of_labels_limits, **kwargs):
        inputs = self.taskDataset.inputs
        targets = self.taskDataset.targets
        if dic_of_labels_limits:
            inputs, targets = self.sample_labels(self.taskDataset, dic_of_labels_limits)

        x_train, x_val, y_train, y_val = \
            train_test_split(inputs, targets,
                             test_size=test_size) \
                if test_size > 0 else (inputs, [], targets, [])
        self.extraction_method.scale_fit(x_train)
        x_train, y_train = self.extraction_method.prepare_inputs_targets(x_train, y_train, **kwargs)
        self.trainTaskDataset = TaskDataset(inputs=x_train, targets=y_train,
                                            name=self.taskDataset.task.name + "_train",
                                            labels=self.taskDataset.task.output_labels,
                                            extraction_method=self.extraction_method,
                                            base_path=self.get_base_path(),
                                            output_module=self.taskDataset.task.output_module,
                                            index_mode=self.index_mode)
        if test_size > 0:
            x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
            self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               extraction_method=self.extraction_method,
                                               base_path=self.get_base_path(),
                                               output_module=self.taskDataset.task.output_module,
                                               index_mode=self.index_mode)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        pass
