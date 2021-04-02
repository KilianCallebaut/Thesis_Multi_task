import torch

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SS import DCASE2017_SS

from DataReaders.ExtractionMethod import Mfcc, MelSpectrogram
from DataReaders.FSDKaggle2018 import FSDKaggle2018
from DataReaders.Ravdess import Ravdess
from DataReaders.SpeechCommands import SpeechCommands
from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Training.Training import Training
from Training.Results import Results

def evaluate_file(filename):
    meta_params = {
        'extraction_method': MelSpectrogram(),
        # 'extraction_method': Mfcc(),
        'batch_size': 1,  # 8,
        'num_epochs': 200,
        'learning_rate': 0.001
    }

    # extraction_params = dict(
    #     nfft=1024,
    #     winlen=0.03,
    #     winstep=0.01,
    #     nfilt=24,
    #     lowfreq=0,
    #     highfreq=None,
    #     preemph=0,
    #     numcep=13,
    #     ceplifter=0,
    #     appendEnergy=False,
    #     winfunc=lambda x: np.ones((x,))
    # )

    extraction_params = dict(
        nfft=1024,
        winlen=0.03,
        winstep=0.01,
        nfilt=96,  # 24,
        lowfreq=0,
        highfreq=None,
        preemph=0,
        winfunc=lambda x: np.ones((x,))
    )

    test_size = 0.2

    preparation_params = dict(
        # test_size=0.2,
        window_size=0,  # 64,
        window_hop=0,  # 32
    )

    preparation_params_val = dict(
        # test_size=0,
        window_size=0,  # 64,
        window_hop=0,  # 32
    )

    model_params = dict(
        hidden_size=64,

        n_hidden=4
    )

    asvspoof = ASVspoof2015(extraction_method=meta_params['extraction_method'],
                            **extraction_params,
                            )
    asvspoof.prepare_taskDatasets(test_size=0, **preparation_params_val)
    asvspoof_t = asvspoof.toTrainTaskDataset()
    asvspoof_e = asvspoof.toValidTaskDataset()
    taskdatasets = [asvspoof_t]
    evaldatasets = [asvspoof_e]
    training_dataset = ConcatTaskDataset([taskdatasets[0]])
    eval_dataset = ConcatTaskDataset([evaldatasets[0]])
    task_list = training_dataset.get_task_list()
    results = Results(training_dataset, 250)
    results.run_name = filename
    model = MultiTaskHardSharingConvolutional(1,
                                              **model_params,
                                              task_list=task_list)
    model = model.cuda()
    with torch.no_grad():
        Training.evaluate(blank_model=model,
                          concat_dataset=eval_dataset,
                          training_results=results,
                          batch_size=1,
                          num_epochs=meta_params['num_epochs'])