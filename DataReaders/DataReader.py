from abc import abstractmethod, ABC

import librosa
import numpy as np
import torch
from python_speech_features import logfbank, mfcc
from scipy import signal
from sklearn.preprocessing import StandardScaler


class DataReader(ABC):
    extractor = None

    @abstractmethod
    def get_path(self):
        pass

    @abstractmethod
    def get_base_path(self):
        pass

    @abstractmethod
    def checkfiles(self, extraction_method):
        pass

    @abstractmethod
    def loadfiles(self):
        pass

    @abstractmethod
    def readfiles(self, extraction_method):
        pass

    @abstractmethod
    def writefiles(self, extraction_method):
        pass

    @abstractmethod
    def calculate_input(self, method, **kwargs):
        pass

    @abstractmethod
    def calculateTaskDataset(self, method, **kwargs):
        pass

    @abstractmethod
    def recalculate_features(self, method, **kwargs):
        pass

    @abstractmethod
    def split_train_test(self, test_size, extraction_method):
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
    def extract_features(self, method, sig_samplerate, **kwargs):
        options = {
            'logbank_summary': self.extract_logbank_summary,
            'logmel': self.extract_logmel,
            'logbank': self.logbank,
            'mfcc': self.mfcc
        }
        return options[method](sig_samplerate=sig_samplerate, **kwargs)

    # Logbank Summary
    def extract_logbank_summary(self, sig_samplerate, **kwargs):
        lb = self.logbank(sig_samplerate, **kwargs)
        return self.logbank_summary(lb)

    def logbank(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
                preemph=0.97):
        return torch.transpose(
            torch.tensor(logfbank(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep,
                                  nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
                                  preemph=preemph)), 0, 1)

    # winfunc = np.hamming
    # ceplifter = 22
    # preemph=0.97
    def mfcc(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
             preemph=0, numcep=13, ceplifter=0, appendEnergy=False, winfunc=lambda x: np.ones((x,))):
        # ( , NUMCEP, )
        return torch.tensor(
            mfcc(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft,
                 lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, numcep=numcep, ceplifter=ceplifter,
                 appendEnergy=appendEnergy, winfunc=winfunc)).T

    def logbank_summary(self, lb):
        # m = torch.min(lb, 1).values
        # mm = torch.max(lb, 1).values
        # s = torch.std(lb, 1)
        # mmm = torch.mean(lb, 1)
        # mmmm = torch.median(lb, 1).values
        # p = torch.tensor(np.percentile(lb, 25, axis=1))
        # pp = torch.tensor(np.percentile(lb, 75, axis=1))
        return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
                            torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
                            torch.tensor(np.percentile(lb, 75, axis=1))])

    # Logmel
    def extract_logmel(self, sig_samplerate, **kwargs):
        sig = sig_samplerate[0]
        fs = sig_samplerate[1]
        if self.extractor is None:
            self.extractor = LogMelExtractor(fs=fs,
                                             **kwargs)
        return self.extractor.transform(sig)

    # TRANSFORMATION
    def convert_list_of_tensors_to_nparray(self, list_of_tensors):
        return np.array([t.numpy() for t in list_of_tensors])

    def convert_nparray_to_list_of_tensors(self, arr):
        return [torch.from_numpy(el) for el in arr]

    def scale_fit(self, inputs, extraction_method):
        self.scalers = {}
        inputs = self.convert_list_of_tensors_to_nparray(inputs)
        if extraction_method in ['logbank_summary', 'logbank']:
            for i in range(inputs.shape[1]):
                self.scalers[i] = StandardScaler()
                self.scalers[i].fit(inputs[:, i, :])

    def scale_transform(self, inputs, extraction_method):
        inputs = self.convert_list_of_tensors_to_nparray(inputs)
        ret = inputs
        if extraction_method in ['logbank_summary', 'logbank']:
            for i in range(inputs.shape[1]):
                ret[:, i, :] = torch.from_numpy(self.scalers[i].transform(inputs[:, i, :]))
        return self.convert_nparray_to_list_of_tensors(ret)

    def scalarization_type(self, extraction_method):
        if extraction_method in []:
            return 'row'
        elif extraction_method in []:
            return 'column'
        elif extraction_method in ['logbank_summary', 'logmel', 'logbank', 'mfcc']:
            return 'general'

    def calculate_scalars(self, inputs, type):
        if type == 'column':
            return self.column_scalars(inputs)
        elif type == 'row':
            return self.row_scalars(inputs)
        elif type == 'general':
            return self.general_scalars(inputs)

    def row_scalars(self, inputs):
        tc = torch.cat(inputs, 1)
        tc_m = torch.mean(tc, 1)
        tc_std = torch.std(tc, 1)
        return tc_m, tc_std

    def column_scalars(self, inputs):
        tc = torch.cat(inputs, 0)
        tc_m = torch.mean(tc, 0)
        tc_std = torch.std(tc, 0)
        return tc_m, tc_std

    def general_scalars(self, inputs):
        tc = torch.cat(inputs, 1)
        tc_m = torch.mean(tc)
        tc_std = torch.std(tc)
        return tc_m, tc_std

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
