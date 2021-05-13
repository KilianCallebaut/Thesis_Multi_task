import os

import joblib
from dcase_util.datasets import TUTAcousticScenes_2017_DevelopmentSet, TUTAcousticScenes_2017_EvaluationSet
from numpy import long
from sklearn.model_selection import train_test_split

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset


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
        self.extraction_method = extraction_method

        print('start DCASE2017 SS')
        if 'object_path' in kwargs:
            self.object_path = kwargs.pop('object_path')
        if self.check_files(extraction_method.name):
            self.read_files(extraction_method.name)
        else:
            self.load_files()
            self.calculate_taskDataset(extraction_method, **kwargs)
            self.write_files(extraction_method.name)

        print('done')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'DCASE2017_SS.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def get_eval_path(self):
        return os.path.join(self.get_eval_base_path(), 'DCASE2017_SS.obj')

    def get_eval_base_path(self):
        return self.object_path.format('eval')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and \
               TaskDataset.check(self.get_eval_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        # MetaDataContainer(filename=)
        self.devdataset = TUTAcousticScenes_2017_DevelopmentSet(
            # data_path='C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system-master\\applications\\data\\',
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
            # data_path=r'C:\\Users\\mrKC1\\PycharmProjects\\Thesis\\ExternalClassifiers\\DCASE2017-baseline-system'
            #           r'-master\\applications\\data\\',
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

    def read_files(self, extraction_method):
        info = joblib.load(self.get_path())
        self.audio_files = info['audio_files']
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)

        info = joblib.load(self.get_eval_path())
        self.audio_files_eval = info['audio_files_eval']
        self.valTaskDataset = TaskDataset([], [], '', [])
        self.valTaskDataset.load(self.get_eval_base_path(), extraction_method)
        print('Reading SS done')

    def write_files(self, extraction_method):
        dict = {'audio_files': self.audio_files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method)

        dict = {'audio_files_eval': self.audio_files_eval}
        joblib.dump(dict, self.get_eval_path())
        self.valTaskDataset.save(self.get_eval_base_path(), extraction_method)

    def calculate_input(self, method, **kwargs):
        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        files = self.audio_files
        inputs = self.calculate_features(files, method, resample_to, **kwargs)
        print("Calculating input done")
        files = self.audio_files_eval
        inputs_val = self.calculate_features(files, method, resample_to, **kwargs)

        return inputs, inputs_val

    def calculate_features(self, files, method, resample_to, **kwargs):
        inputs = []
        perc = 0
        for audio_idx in range(len(files)):
            read_wav = self.load_wav(files[audio_idx], resample_to)
            inputs.append(method.extract_features(read_wav, **kwargs))
            if perc < (audio_idx / len(files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        print("Calculating input done")
        return inputs

    def calculate_taskDataset(self, method, **kwargs):
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

        inputs, inputs_val = self.calculate_input(method, **kwargs)

        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='DCASE2017_SS',
                                       labels=distinct_labels,
                                       output_module='softmax')

        self.valTaskDataset = TaskDataset(inputs=inputs_val,
                                          targets=targets_val,
                                          name='DCASE2017_SS_eval',
                                          labels=distinct_labels,
                                          output_module='softmax'
                                          )

    def recalculate_features(self, method, **kwargs):
        inputs, inputs_val = self.calculate_input(method, **kwargs)
        self.taskDataset.inputs = inputs
        self.valTaskDataset.inputs = inputs_val

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
                                            output_module=self.taskDataset.task.output_module)
        if test_size > 0:
            x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
            self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               output_module=self.taskDataset.task.output_module)

        self.valTaskDataset.inputs, self.valTaskDataset.targets \
            = self.extraction_method.prepare_inputs_targets(self.valTaskDataset.inputs, self.valTaskDataset.targets,
                                                            **kwargs)

    def make_train_test_TaskDatasets(self, x_train, y_train, x_val, y_val, **kwargs):
        self.extraction_method.scale_fit(x_train)
        x_train, y_train = self.extraction_method.prepare_inputs_targets(x_train, y_train, **kwargs)
        self.trainTaskDataset = TaskDataset(inputs=x_train, targets=y_train,
                                            name=self.taskDataset.task.name + "_train",
                                            labels=self.taskDataset.task.output_labels,
                                            output_module=self.taskDataset.task.output_module)
        x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
        self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                           name=self.taskDataset.task.name + "_test",
                                           labels=self.taskDataset.task.output_labels,
                                           output_module=self.taskDataset.task.output_module)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        return self.valTaskDataset
