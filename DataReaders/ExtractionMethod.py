import math
import os
import statistics
from abc import abstractmethod, ABC

import numpy as np
import torch
import joblib
import pickle
from python_speech_features import logfbank, mfcc, fbank
from sklearn.preprocessing import StandardScaler


class ExtractionMethod(ABC):

    def __init__(self):
        self.scalers = {}
        self.name = ''

    @abstractmethod
    def extract_features(self, sig_samplerate, **kwargs):
        pass

    @abstractmethod
    def scale_fit(self, inputs):
        pass

    @abstractmethod
    def scale_transform(self, inputs):
        pass

    @abstractmethod
    def prepare_inputs(self, inputs, **kwargs):
        pass

    # EXTRACTION
    def logbank(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
                preemph=0.97):
        return torch.transpose(
            torch.tensor(logfbank(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep,
                                  nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
                                  preemph=preemph)), 0, 1)

    # SCALING
    def convert_list_of_tensors_to_nparray(self, list_of_tensors):
        return np.array([t.numpy() for t in list_of_tensors])

    def convert_nparray_to_list_of_tensors(self, arr):
        return [torch.from_numpy(el) for el in arr]

    # scale fitters
    def scale_fit_3D_2nddim(self, inputs):
        """
        Sets the scalers based on the input per the second dimension in a 3D matrix
        :param inputs: ndarray
        """
        for i in range(inputs.shape[1]):
            self.scalers[i] = StandardScaler()
            self.scalers[i].fit(inputs[:, i, :])

    def scale_fit_2D(self, inputs):
        self.scalers[0] = StandardScaler()
        self.scalers[0].fit(inputs)

    # scale transformers
    def scale_transform_3D_2nddim(self, inputs):
        """
        Scales an input for both feature dimensions in a 3D matrix
        :param inputs: ndarray
        :return: transformed input
        """
        ret = inputs
        for i in range(inputs.shape[1]):
            ret[:, i, :] = self.scalers[i].transform(inputs[:, i, :])
        return ret

    def scale_transform_2D(self, inputs):
        """
        Scales an input for 2D matrix
        :param inputs: ndarray
        :return: transformed input
        """
        ret = self.scalers[0].transform(inputs)
        return ret

    # WINDOWING

    def window_inputs(self, inputs, targets, window_size=64, window_hop=32):
        windowed_inputs = []
        windowed_targets = []
        start_frame = window_size

        for inp_idx in range(len(inputs)):
            inp = inputs[inp_idx]
            end_frame = start_frame + window_hop * math.floor((float(inp.shape[0] - start_frame) / window_hop))
            for frame_idx in range(start_frame, end_frame + 1, window_hop):
                window = inp[frame_idx - window_size:frame_idx, :]
                assert window.shape == (window_size, inp.shape[1])
                windowed_inputs.append(window)
                windowed_targets.append(targets[inp_idx])

            if start_frame > inp.shape[0]:
                window = torch.vstack([inp, torch.zeros(start_frame - inp.shape[0], inp.shape[1])])
                windowed_inputs.append(window)
                windowed_targets.append(targets[inp_idx])
            elif end_frame < inp.shape[0]:
                window = inp[inp.shape[0] - window_size:inp.shape[0], :]
                windowed_inputs.append(window)
                windowed_targets.append(targets[inp_idx])

        assert len(windowed_inputs) == len(windowed_targets)
        return windowed_inputs, windowed_targets

    def frame_inputs(self, inputs, window_size):
        framed_inputs = []

        for inp_idx in range(len(inputs)):
            inp = inputs[inp_idx]
            if window_size > len(inp):
                frame = torch.vstack([inp, torch.zeros((window_size - inp.shape[0], inp.shape[1]))])
            else:
                frame = inp[0:window_size]
            framed_inputs.append(frame)
        return framed_inputs

class LogbankSummary(ExtractionMethod):

    def __init__(self):
        super().__init__()
        self.name = 'logbank_summary'

    def extract_features(self, sig_samplerate, **kwargs):
        lb = self.logbank(sig_samplerate, **kwargs)
        return self.logbank_summary(lb)

    def logbank_summary(self, lb):
        return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
                            torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
                            torch.tensor(np.percentile(lb, 75, axis=1))])

    def scale_fit(self, inputs):
        inputs = self.convert_list_of_tensors_to_nparray(inputs)
        self.scale_fit_3D_2nddim(inputs)

    def scale_transform(self, inputs):
        inputs = self.convert_list_of_tensors_to_nparray(inputs)
        ret = self.scale_transform_3D_2nddim(inputs)
        return self.convert_nparray_to_list_of_tensors(ret)

    def prepare_inputs(self, inputs, **kwargs):
        # inputs = self.scale_transform(inputs)
        return inputs


class Mfcc(ExtractionMethod):

    def __init__(self):
        super().__init__()
        self.name = 'mfcc'

    def extract_features(self, sig_samplerate, **kwargs):
        return self.mfcc(sig_samplerate, **kwargs)

    # winfunc = np.hamming
    # ceplifter = 22
    # preemph=0.97
    def mfcc(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
             preemph=0, numcep=13, ceplifter=0, appendEnergy=False, winfunc=lambda x: np.ones((x,))):
        # ( NUMFRAMES, NUMCEP)
        return torch.tensor(
            mfcc(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep, nfilt=nfilt, nfft=nfft,
                 lowfreq=lowfreq, highfreq=highfreq, preemph=preemph, numcep=numcep, ceplifter=ceplifter,
                 appendEnergy=appendEnergy, winfunc=winfunc))

    def scale_fit(self, inputs):
        inputs = np.concatenate(inputs)
        self.scale_fit_2D(inputs)

    def scale_transform(self, inputs):
        # ret = self.convert_list_of_tensors_to_nparray(inputs)
        ret = []
        for i in range(len(inputs)):
            ret.append(self.scale_transform_2D(inputs[i].numpy()))
        return self.convert_nparray_to_list_of_tensors(ret)

    def prepare_inputs(self, inputs, **kwargs):
        cum_sizes = [len(i) for i in inputs]

        if 'window_size' in kwargs:
            inputs = self.frame_inputs(inputs, kwargs.get('window_size'))
        elif max(cum_sizes) != min(cum_sizes):
            median = statistics.median(cum_sizes)
            inputs = self.frame_inputs(inputs, median)
        # inputs = self.scale_transform(inputs)

        return inputs


class MelSpectrogram(ExtractionMethod):

    def __init__(self):
        super().__init__()
        self.name = 'MelSpectrogram'

    def extract_features(self, sig_samplerate, **kwargs):
        """
        (numframes=1 + int(math.ceil((1.0*sig_len - frame_len=winlen*samplerate)/frame_step=winstep*samplerate)), nfilt)
        for winlen=0.03, winstep=0.01, sr=44100:
            numframes= 1 + math.ceil((sig_len - 1323.0)/441)
        for 1 sec:
            numframes = 1 + math.ceil((44100 - 1323)/441)
        """
        return torch.tensor(fbank(sig_samplerate[0], sig_samplerate[1],
                                  **kwargs)[0])

    def scale_fit(self, inputs):
        inputs = np.concatenate(inputs)
        self.scale_fit_2D(inputs)

    def scale_transform(self, inputs):
        # ret = self.convert_list_of_tensors_to_nparray(inputs)
        ret = []
        for i in range(len(inputs)):
            ret.append(self.scale_transform_2D(inputs[i].numpy()))
        return self.convert_nparray_to_list_of_tensors(ret)

    def prepare_inputs(self, inputs, **kwargs):
        cum_sizes = [len(i) for i in inputs]
        if 'window_size' in kwargs:
            inputs = self.frame_inputs(inputs, kwargs.get('window_size'))
        elif max(cum_sizes) != min(cum_sizes):
            median = statistics.median(cum_sizes)
            inputs = self.frame_inputs(inputs, int(median))
        # inputs = self.scale_transform(inputs)

        return inputs


extract_options = {
    MelSpectrogram().name: MelSpectrogram(),
    Mfcc().name: Mfcc(),
    LogbankSummary().name: LogbankSummary()
}
