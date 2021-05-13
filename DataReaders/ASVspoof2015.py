import os

import joblib
import pandas as pd
from numpy import long
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


class ASVspoof2015(DataReader):
    Wav_folder = r"F:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853\wav"
    Label_folder = r"F:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853\Joint_ASV_CM_protocol"
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ASVspoof2015_{}"

    # object_path = r"E:\Thesis_Results\Data_Readers\ASVspoof2015_{}"

    def __init__(self, extraction_method, **kwargs):
        self.extraction_method = extraction_method

        print('start asvspoof2015')
        if 'object_path' in kwargs:
            self.object_path = kwargs.pop('object_path')
        if self.check_files(extraction_method=self.extraction_method.name):
            self.read_files(self.extraction_method.name)
        else:
            self.load_files()
            self.calculate_taskDataset(self.extraction_method, **kwargs)
            self.write_files(self.extraction_method.name)

        print('done')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'ASVspoof2015.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def get_eval_base_path(self):
        return self.object_path.format('eval')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) \
               and os.path.isfile(self.get_path()) and TaskDataset.check(self.get_eval_base_path(), extraction_method)

    def load_files(self):
        self.truths = pd.read_csv(os.path.join(self.Label_folder, 'ASV_male_development.ndx'), sep=' ', header=None,
                                  names=['folder', 'file', 'method', 'source'])
        truths_female = pd.read_csv(os.path.join(self.Label_folder, 'ASV_female_development.ndx'), sep=' ',
                                    header=None,
                                    names=['folder', 'file', 'method', 'source'])
        self.truths.append(truths_female)

        self.truths_val = pd.read_csv(os.path.join(self.Label_folder, 'ASV_male_evaluation.ndx'), sep=' ', header=None,
                                      names=['folder', 'file', 'method', 'source'])
        truths_female = pd.read_csv(os.path.join(self.Label_folder, 'ASV_female_evaluation.ndx'), sep=' ',
                                    header=None,
                                    names=['folder', 'file', 'method', 'source'])
        self.truths_val.append(truths_female)

        self.truths = self.truths[(self.truths.method == 'genuine')]
        self.truths.sort_values(['folder', 'file'], inplace=True)

        self.truths_val = self.truths_val[(self.truths_val.method == 'genuine')]
        self.truths_val.sort_values(['folder', 'file'], inplace=True)

        self.files = [os.path.join(self.Wav_folder, x[0], x[1]) for x in self.truths.to_numpy()]
        self.files_val = [os.path.join(self.Wav_folder, x[0], x[1]) for x in self.truths_val.to_numpy()]

    def read_files(self, extraction_method):
        info = joblib.load(self.get_path())
        self.files = info['files']
        self.truths = info['truths']
        self.files_val = info['files_val']
        self.truths_val = info['truths_val']
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)

        self.valTaskDataset = TaskDataset([], [], '', [])
        self.valTaskDataset.load(self.get_eval_base_path(), extraction_method)

    def write_files(self, extraction_method):
        dict = {'files': self.files,
                'truths': self.truths,
                'files_val': self.files_val,
                'truths_val': self.truths_val
                }
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method)
        self.valTaskDataset.save(self.get_eval_base_path(), extraction_method)

    # which = develop, evaluation
    def calculate_input(self, method, **kwargs):
        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        print('training')
        perc = 0
        inputs = []
        for audio_idx in range(len(self.files)):
            read_wav = self.load_wav(self.files[audio_idx] + '.wav', resample_to)
            inputs.append(method.extract_features(read_wav, **kwargs))
            if perc < (audio_idx / len(self.files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

        print('validation')
        perc = 0
        inputs_val = []
        for audio_idx in range(len(self.files_val)):
            read_wav = self.load_wav(self.files_val[audio_idx] + '.wav', resample_to)
            inputs_val.append(method.extract_features(read_wav, **kwargs))
            if perc < (audio_idx / len(self.files_val)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

        # inputs = self.pad(inputs, max(self.max_length(inputs), self.max_length(inputs_val)))
        # inputs_val = self.pad(inputs, max(self.max_length(inputs), self.max_length(inputs_val)))
        return inputs, inputs_val

    def calculate_taskDataset(self, method, **kwargs):
        distinct_labels = self.truths.folder.unique()
        distinct_labels.sort()
        targets = []
        for f in self.truths.folder.to_numpy():
            target = [long(distinct_labels[label_id] == f) for label_id in range(len(distinct_labels))]
            targets.append(target)

        targets_val = []
        for f in self.truths_val.folder.to_numpy():
            target = [long(distinct_labels[label_id] == f) for label_id in range(len(distinct_labels))]
            targets_val.append(target)

        inputs, inputs_val = self.calculate_input(method, **kwargs)

        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='ASVspoof2015',
                                       labels=distinct_labels,
                                       output_module='softmax')

        self.valTaskDataset = TaskDataset(inputs=inputs_val,
                                          targets=targets_val,
                                          name='ASVspoof2015_val',
                                          labels=distinct_labels,
                                          output_module='softmax')

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
            train_test_split(inputs, targets, test_size=test_size) \
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

        self.valTaskDataset.inputs, self.valTaskDataset.targets = self.extraction_method.prepare_inputs_targets(
            self.valTaskDataset.inputs, self.valTaskDataset.targets, **kwargs)

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
