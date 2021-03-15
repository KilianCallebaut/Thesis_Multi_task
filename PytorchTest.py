import torch
import torchaudio
import requests
import matplotlib.pyplot as plt


def open_and_plot_audio(url, filename):
    # Opening file
    # url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    # filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"

    r = requests.get(url)

    with open(filename, 'wb') as f:
        f.write(r.content)

    waveform, sample_rate = torchaudio.load(filename)

    # transform(waveform)
    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    plt.figure()
    plt.plot(waveform.t().numpy())
    # plt.show()
    return waveform, sample_rate


def transform(waveform):
    # transformations:
    # Resample
    # Spectogram
    # GriffinLim
    # ComputeDeltas
    # ComplexNorm
    # MelScale
    # AmplitudeToDB
    # MFCC
    # MelSpectrogram
    # MuLawEncoding
    # MuLawDecoding
    # TimeStretch
    # FrequencyMasking
    # TimeMasking
    specgram = torchaudio.transforms.MelSpectrogram()(waveform)

    print("Shape of spectrogram: {}".format(specgram.size()))

    plt.figure()
    plt.imshow(specgram.log2()[0, :, :].numpy(), cmap='gray')
    return specgram


def resample(waveform, sample_rate):
    new_sample_rate = sample_rate / 10.0
    channel = 0
    transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))
    (waveform[channel, :].view(1, -1))

    print("Shape of transformed waveform: {}".format(transformed.size()))
    plt.figure()
    plt.plot(transformed[0, :].numpy())
    return transformed


def all():
    url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    r = requests.get(url)

    with open('steam-train-whistle-daniel_simon-converted-from-mp3.wav', 'wb') as f:
        f.write(r.content)

    filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    waveform, sample_rate = torchaudio.load(filename)

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    plt.figure()
    plt.plot(waveform.t().numpy())

    #Spectrogram
    specgram = torchaudio.transforms.Spectrogram()(waveform)

    print("Shape of spectrogram: {}".format(specgram.size()))

    plt.figure()
    plt.imshow(specgram.log2()[0, :, :].numpy(), cmap='gray')

    #Melspec
    specgram = torchaudio.transforms.MelSpectrogram()(waveform)

    print("Shape of spectrogram: {}".format(specgram.size()))

    plt.figure()
    p = plt.imshow(specgram.log2()[0, :, :].detach().numpy(), cmap='gray')

    #resample
    new_sample_rate = sample_rate / 10

    # Since Resample applies to a single channel, we resample first channel here
    channel = 0
    transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))

    print("Shape of transformed waveform: {}".format(transformed.size()))
    plt.figure()
    plt.plot(transformed[0, :].numpy())

def runtest():
    url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"
    waveform, sample_rate = open_and_plot_audio(url, filename)
    waveform_transformed = transform(waveform)
    resample(waveform, sample_rate)
    plt.show()
