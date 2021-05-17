import random
from abc import abstractmethod, ABC

import librosa
import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import KFold


class DataReader(ABC):
    extractor = None

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
    def calculate_input(self,  **kwargs):
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

    def k_folds(self, dic_of_labels_limits, taskDataset):
        # Examples
        # --------
        # >> > import numpy as np
        # >> > from sklearn.model_selection import KFold
        # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >> > y = np.array([1, 2, 3, 4])
        # >> > kf = KFold(n_splits=2)
        # >> > kf.get_n_splits(X)
        # 2
        # >> > print(kf)
        # KFold(n_splits=2, random_state=None, shuffle=False)
        # >> > for train_index, test_index in kf.split(X):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # ...
        # X_train, X_test = X[train_index], X[test_index]
        # ...
        # y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [2 3]
        # TEST: [0 1]
        # TRAIN: [0 1]
        # TEST: [2 3]

        kf = KFold(n_splits=5)
        inputs = taskDataset.inputs
        if dic_of_labels_limits:
            inputs, _ = self.sample_labels(taskDataset, dic_of_labels_limits)
        kf.split(inputs)

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
        if resample_to is not None:
            sig, fs = librosa.load(loc, sr=resample_to, mono=False, dtype=np.float32)
        else:
            sig, fs = librosa.load(loc, mono=False, dtype=np.float32)
        return sig, fs

    # FEATURE EXTRACTION
    # def extract_features(self, method, sig_samplerate, **kwargs):
    #     options = {
    #         'logbank_summary': self.extract_logbank_summary,
    #         'logmel': self.extract_logmel,
    #         'logbank': self.logbank,
    #         'mfcc': self.mfcc
    #     }
    #     return options[method](sig_samplerate=sig_samplerate, **kwargs)

    # Logbank Summary
    # def extract_logbank_summary(self, sig_samplerate, **kwargs):
    #     lb = self.logbank(sig_samplerate, **kwargs)
    #     return self.logbank_summary(lb)

    # def logbank(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
    #             preemph=0.97):
    #     return torch.transpose(
    #         torch.tensor(logfbank(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep,
    #                               nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
    #                               preemph=preemph)), 0, 1)

    # winfunc = np.hamming
    # ceplifter = 22
    # preemph=0.97
    # def mfcc(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
    #          preemph=0, numcep=13, ceplifter=0, appendEnergy=False, winfunc=lambda x: np.ones((x,))):
    #     # ( , NUMCEP, )
    #     return torch.tensor(
    #         mfcc(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft,
    #              lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, numcep=numcep, ceplifter=ceplifter,
    #              appendEnergy=appendEnergy, winfunc=winfunc)).T

    # def logbank_summary(self, lb):
    #     return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
    #                         torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
    #                         torch.tensor(np.percentile(lb, 75, axis=1))])

    # Logmel
    # def extract_logmel(self, sig_samplerate, **kwargs):
    #     sig = sig_samplerate[0]
    #     fs = sig_samplerate[1]
    #     if self.extractor is None:
    #         self.extractor = LogMelExtractor(fs=fs,
    #                                          **kwargs)
    #     return self.extractor.transform(sig)

    # # Padding
    # def max_length(self, inputs, dim=1):
    #     return max([inp.shape[dim] for inp in inputs])
    #
    # def pad(self, list_of_tensors, max_length, dim=1):
    #     return [torch.cat([l, torch.zeros(l.shape[0], max_length - l.shape[1])], 1) for l in list_of_tensors]

    # Context window
    # def window_inputs(self, inputs, targets, window_size=64, window_hop=30):
    #     windowed_inputs = []
    #     windowed_targets = []
    #     start_frame = window_size
    #
    #     for inp_idx in range(len(inputs)):
    #         inp = inputs[inp_idx]
    #         end_frame = window_hop * math.floor(float(inp.shape[1]) / window_hop)
    #         for frame_idx in range(start_frame, end_frame, window_hop):
    #             window = inp[:, frame_idx - window_size:frame_idx]
    #             assert window.shape == (inp.shape[0], window_size)
    #             windowed_inputs.append(window)
    #             windowed_targets.append(targets[inp_idx])
    #
    #     assert len(windowed_inputs) == len(windowed_targets)
    #     return windowed_inputs, windowed_targets

    # TRANSFORMATION
    # def convert_list_of_tensors_to_nparray(self, list_of_tensors):
    #     return np.array([t.numpy() for t in list_of_tensors])
    #
    # def convert_nparray_to_list_of_tensors(self, arr):
    #     return [torch.from_numpy(el) for el in arr]

    # def scale_fit(self, inputs, extraction_method):
    #     self.scalers = {}
    #     inputs = self.convert_list_of_tensors_to_nparray(inputs)
    #     if extraction_method in ['logbank_summary', 'mfcc']:
    #         for i in range(inputs.shape[1]):
    #             self.scalers[i] = StandardScaler()
    #             self.scalers[i].fit(inputs[:, i, :])
    #
    # def scale_transform(self, inputs, extraction_method):
    #     inputs = self.convert_list_of_tensors_to_nparray(inputs)
    #     ret = inputs
    #     if extraction_method in ['logbank_summary', 'logbank']:
    #         for i in range(inputs.shape[1]):
    #             ret[:, i, :] = torch.from_numpy(self.scalers[i].transform(inputs[:, i, :]))
    #     return self.convert_nparray_to_list_of_tensors(ret)

    # def standardize_input(self, inputs, means, stds, type):
    #     if type == 'column':
    #         return self.column_standardize(inputs, means, stds)
    #     elif type == 'row':
    #         return self.row_standardize(inputs, means, stds)
    #     elif type == 'general':
    #         return self.general_standardize(inputs, means, stds)
    #
    # def column_standardize(self, inputs, means, stds):
    #     ret = []
    #     for inp in inputs:
    #         r = inp.transpose(0, 1)
    #         r = torch.stack([(inp[i] - means[i]) / stds[i] for i in range(len(r))])
    #         r = r.transpose(0, 1)
    #         ret.append(r)
    #     return ret
    #
    # def row_standardize(self, inputs, means, stds):
    #     ret = []
    #     for inp in inputs:
    #         r = torch.stack([(inp[i] - means[i]) / stds[i] for i in range(len(inp))])
    #         ret.append(r)
    #     return ret
    #
    # def general_standardize(self, inputs, mean, std):
    #     ret = []
    #     for inp in inputs:
    #         r = (inp - mean)/std
    #         ret.append(r)
    #     return ret

    # # General standardization CNN
    # # # TRANSFORMATION
    # def calculate_scalars(self, inputs):
    #     tc = torch.cat(inputs, 1)
    #     tc_m = torch.mean(tc)
    #     tc_std = torch.std(tc)
    #     return tc_m, tc_std
    #
    # def standardize_input(self, inputs, means, stds):
    #     ret = []
    #     for inp in inputs:
    #         r = (inp - means)/stds
    #         ret.append(r)
    #     return ret
    #


class LogMelExtractor:
    # nfft=1024, winlen=None, hop_length = 320, mel_bins=96, lowfreq=50
    def __init__(self, fs, nfft=512, winlen=300, winstep=100, nfilt=24, window='hann', lowfreq=0):
        self.nfft = nfft
        self.win_length = winlen
        self.hop_length = winstep
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=nfilt,
                                        fmin=lowfreq)

    def transform(self, audio):
        channel_num = audio.shape[0]
        feature_logmel = []

        for n in range(channel_num):
            S = np.abs(librosa.stft(y=audio[n],
                                    n_fft=self.nfft,
                                    win_length=self.win_length,
                                    hop_length=self.hop_length,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect')) ** 2

            S_mel = np.dot(self.melW, S).T
            S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
            S_logmel = np.expand_dims(S_logmel, axis=0)
            feature_logmel.append(S_logmel)

        feature_logmel = np.concatenate(feature_logmel, axis=0)

        return torch.tensor(feature_logmel)
