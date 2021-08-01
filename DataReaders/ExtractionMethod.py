import math
import statistics
from abc import abstractmethod, ABC
from typing import List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from python_speech_features import logfbank, mfcc, fbank
from sklearn.preprocessing import StandardScaler


class ExtractionMethod(ABC):

    def __init__(self, name, preparation_params=None, extraction_params=None):
        if extraction_params is None:
            extraction_params = dict()
        if preparation_params is None:
            preparation_params = dict()
        self.scalers = {}
        self.name = name
        self.extraction_params = extraction_params
        self.preparation_params = preparation_params
        self.prep_calcs = {}

    @abstractmethod
    def extract_features(self, sig_samplerate) -> torch.tensor:
        pass

    @abstractmethod
    def partial_scale_fit(self, input_tensor):
        pass

    @abstractmethod
    def scale_transform(self, input_tensor) -> torch.tensor:
        pass

    @abstractmethod
    def inverse_scale_transform(self, input_tensor) -> torch.tensor:
        pass

    @abstractmethod
    def prepare_fit(self, input_tensor: torch.tensor):
        pass

    @abstractmethod
    def prepare_input(self, input_tensor: torch.tensor) -> List[torch.tensor]:
        pass

    # EXTRACTION
    def logbank(self, sig_samplerate, winlen=0.03, winstep=0.01, nfilt=24, nfft=512, lowfreq=0, highfreq=None,
                preemph=0.97):
        return torch.transpose(
            torch.tensor(logfbank(sig_samplerate[0], sig_samplerate[1], winlen=winlen, winstep=winstep,
                                  nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
                                  preemph=preemph)), 0, 1)

    # scale fitters
    def partial_scale_fit_1st_dim(self, input_tensor):
        """
        Sets the scalers based on the input per row
        :param input_tensor: ndarray
        """
        if not self.scalers:
            for i in range(input_tensor.shape[0]):
                self.scalers[i] = StandardScaler()
        for i in range(input_tensor.shape[0]):
            self.scalers[i].partial_fit(input_tensor[i, :].reshape(1, -1))

    def partial_scale_fit_2D(self, input_tensor):
        """
        Adds the current input to the scaling calculation
        :param input_tensor:
        :return:
        """
        if not self.scalers[0]:
            self.scalers[0] = StandardScaler()
        self.scalers[0].partial_fit(input_tensor)

    # scale transformers
    def scale_transform_1st_dim(self, input_tensor):
        """
        Scales an input for the first axis in a tensor
        :param inputs: ndarray
        :return: transformed input
        """
        for i in range(input_tensor.shape[0]):
            input_tensor[i, :] = self.scalers[i].transform(input_tensor[i, :].reshape(1, -1))
        return input_tensor

    def inverse_scale_transform_1st_dim(self, input_tensor):
        """
        Undoes a performed scaling
        :param input: ndarray
        :return: original input
        """
        for i in range(input_tensor.shape[0]):
            input_tensor[i, :] = self.scalers[i].inverse_transform(input_tensor[i, :].reshape(1, -1))
        return input_tensor

    def scale_transform_2D(self, input_tensor):
        """
        Scales an input for 2D matrix
        :param input_tensor: ndarray
        :return: transformed input
        """
        return self.scalers[0].transform(input_tensor)

    def inverse_scale_transform_2D(self, input_tensor):
        """
        Undoes a performed scaling
        :param input_tensor: ndarray
        :return: original input
        """
        return self.scalers[0].inverse_transform(input_tensor)

    def scale_reset(self):
        for s in self.scalers:
            s._reset()

    # Preparation
    def median_window_size_fit(self, input_tensor):
        if not 'cum_sizes' in self.prep_calcs:
            self.prep_calcs['cum_sizes'] = []
        self.prep_calcs['cum_sizes'].append(len(input_tensor))
        self.preparation_params['window_size'] = statistics.median(self.prep_calcs['cum_sizes'])

    def window_inputs(self, input_tensor, window_size=64, window_hop=32):
        windowed_inputs = []
        start_frame = window_size

        end_frame = start_frame + window_hop * math.floor((float(input_tensor.shape[0] - start_frame) / window_hop))
        for frame_idx in range(start_frame, end_frame + 1, window_hop):
            window = input_tensor[frame_idx - window_size:frame_idx, :]
            assert window.shape == (window_size, input_tensor.shape[1])
            windowed_inputs.append(window)

        if start_frame > input_tensor.shape[0]:
            window = torch.vstack(
                [input_tensor, torch.zeros(start_frame - input_tensor.shape[0], input_tensor.shape[1])])
            windowed_inputs.append(window)
        elif end_frame < input_tensor.shape[0]:
            window = input_tensor[input_tensor.shape[0] - window_size:input_tensor.shape[0], :]
            windowed_inputs.append(window)

        return windowed_inputs

    def frame_inputs(self, input_tensor, window_size):
        if window_size > len(input_tensor):
            frame = torch.vstack(
                [input_tensor, torch.zeros((window_size - input_tensor.shape[0], input_tensor.shape[1]))])
        else:
            frame = input_tensor[0:window_size]
        return frame


class LogbankSummary(ExtractionMethod):

    def __init__(self, preparation_params=None, extraction_params=None):
        super().__init__(name='logbank_summary', preparation_params=preparation_params,
                         extraction_params=extraction_params)

    def extract_features(self, sig_samplerate):
        lb = self.logbank(sig_samplerate, **self.extraction_params)
        return self.logbank_summary(lb)

    def logbank_summary(self, lb):
        return torch.stack([torch.min(lb, 1).values, torch.max(lb, 1).values, torch.std(lb, 1), torch.mean(lb, 1),
                            torch.median(lb, 1).values, torch.tensor(np.percentile(lb, 25, axis=1)),
                            torch.tensor(np.percentile(lb, 75, axis=1))])

    def partial_scale_fit(self, input_tensor):
        self.partial_scale_fit_1st_dim(input_tensor)

    def scale_transform(self, input_tensor: torch.tensor):
        return torch.from_numpy(self.scale_transform_1st_dim(input_tensor))

    def inverse_scale_transform(self, input_tensor):
        return torch.from_numpy(self.inverse_scale_transform_1st_dim(input_tensor))

    def prepare_input(self, input_tensor):
        return input_tensor


class Mfcc(ExtractionMethod):

    def __init__(self, preparation_params=None, extraction_params=None):
        super().__init__(name='mfcc', preparation_params=preparation_params, extraction_params=extraction_params)

    def extract_features(self, sig_samplerate):
        return self.mfcc(sig_samplerate, **self.extraction_params)

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

    def partial_scale_fit(self, input_tensor):
        self.partial_scale_fit_2D(input_tensor)

    def scale_transform(self, input_tensor):
        return torch.from_numpy(self.scale_transform_2D(input_tensor.numpy()))

    def inverse_scale_transform(self, input_tensor):
        return torch.from_numpy(self.inverse_scale_transform_2D(input_tensor))

    def prepare_fit(self, input_tensor: torch.tensor):
        self.median_window_size_fit(input_tensor)

    def prepare_input(self, input_tensor):
        return self.frame_inputs(input_tensor, **self.preparation_params)


class MelSpectrogram(ExtractionMethod):

    def __init__(self, preparation_params=None, extraction_params=None):
        super().__init__(name='MelSpectrogram', preparation_params=preparation_params,
                         extraction_params=extraction_params)

    def extract_features(self, sig_samplerate):
        """
        (numframes=1 + int(math.ceil((1.0*sig_len - frame_len=winlen*samplerate)/frame_step=winstep*samplerate)), nfilt)
        for winlen=0.03, winstep=0.01, sr=44100:
            numframes= 1 + math.ceil((sig_len - 1323.0)/441)
        for 1 sec:
            numframes = 1 + math.ceil((44100 - 1323)/441)
        """
        return torch.tensor(fbank(sig_samplerate[0], sig_samplerate[1],
                                  **self.extraction_params)[0])

    def partial_scale_fit(self, input_tensor):
        self.partial_scale_fit_2D(input_tensor)

    def scale_transform(self, input_tensor):
        return torch.from_numpy(self.scale_transform_2D(input_tensor.numpy()))

    def inverse_scale_transform(self, input_tensor):
        return torch.from_numpy(self.inverse_scale_transform_2D(input_tensor.numpy()))

    def prepare_fit(self, input_tensor: torch.tensor):
        self.median_window_size_fit(input_tensor)

    def prepare_input(self, input_tensor):
        return self.frame_inputs(input_tensor, **self.preparation_params)


class LibMelSpectrogram(ExtractionMethod):

    def __init__(self, preparation_params=None, extraction_params=None):
        super().__init__(name='LibMelSpectrogram', preparation_params=preparation_params,
                         extraction_params=extraction_params)

    def extract_features(self, sig_samplerate):
        feature = librosa.feature.melspectrogram(y=sig_samplerate[0], sr=sig_samplerate[1], **self.extraction_params)
        feature = librosa.power_to_db(feature, ref=np.max)
        feature = librosa.util.normalize(feature)
        # plot_feature(feature, 16000 )
        return torch.tensor(feature).T

    def partial_scale_fit(self, input_tensor):
        self.partial_scale_fit_2D(input_tensor)

    def scale_transform(self, input_tensor):
        return torch.from_numpy(self.scale_transform_2D(input_tensor.numpy()))

    def inverse_scale_transform(self, input_tensor):
        return torch.from_numpy(self.inverse_scale_transform_2D(input_tensor.numpy()))

    def prepare_fit(self, input_tensor: torch.tensor):
        self.median_window_size_fit(input_tensor)

    def prepare_input(self, input_tensor):
        return self.frame_inputs(input_tensor, **self.preparation_params)


extract_options = {
    MelSpectrogram().name: MelSpectrogram,
    Mfcc().name: Mfcc,
    LogbankSummary().name: LogbankSummary,
    LibMelSpectrogram().name: LibMelSpectrogram
}


def plot_feature(S, sr):
    fig, ax = plt.subplots()
    # S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S, x_axis='time',
                                   y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Spectrogram')
    plt.show()
