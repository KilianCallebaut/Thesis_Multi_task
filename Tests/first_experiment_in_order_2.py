from __future__ import print_function, division

import itertools
import sys

import torch

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SE import DCASE2017_SE
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.ExtractionMethod import MelSpectrogramExtractionMethod
from DataReaders.FSDKaggle2018 import FSDKaggle2018
from DataReaders.Ravdess import Ravdess
from DataReaders.SpeechCommands import SpeechCommands
from MultiTask.AdaptedBaselineCnn import BaselineCnn
from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from Tests.config_reader import *
from Training.Training import Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'E'
data_base = r'E:\Thesis_Results\Data_Readers'


def main(argv):
    print('--------------------------------------------------')
    print('test loop')
    print('--------------------------------------------------')

    extraction_params = {
        "extraction_params": {
            "n_fft": 2048,
            "hop_length": 256,
            "n_mels": 128,
            "window": "hann"
        },
    }

    meta_params = {
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 0.001
    }

    mtmf = MultiTaskModelFactory()
    mtmf.add_modelclass(MultiTaskHardSharing)
    mtmf.add_static_model_parameters(MultiTaskHardSharing.__name__,
                                     **read_config('model_params_cnn'),
                                     input_channels=1)
    mtmf.add_modelclass(MultiTaskHardSharingConvolutional)
    mtmf.add_static_model_parameters(MultiTaskHardSharingConvolutional.__name__,
                                     **read_config('model_params_dnn'))

    csc = ConcatTrainingSetCreator(random_state=123, nr_runs=5)
    csc.add_data_reader(ASVspoof2015(object_path=os.path.join(data_base, 'ASVspoof2015_{}'), index_mode=True))
    csc.add_data_reader(
        ChenAudiosetDataset(object_path=os.path.join(data_base, 'ChenAudiosetDataset'), index_mode=True))
    csc.add_data_reader(DCASE2017_SS(object_path=os.path.join(data_base, 'DCASE2017_SS_{}'), index_mode=True))
    csc.add_data_reader(DCASE2017_SE(object_path=os.path.join(data_base, 'DCASE2017_SE_{}'), index_mode=True))
    csc.add_data_reader(FSDKaggle2018(object_path=os.path.join(data_base, 'FSDKaggle2018_{}'), index_mode=True))
    csc.add_data_reader(Ravdess(object_path=os.path.join(data_base, 'Ravdess'), index_mode=True))
    csc.add_data_reader(SpeechCommands(object_path=os.path.join(data_base, 'SpeechCommands_{}'), index_mode=True))

    csc.add_sample_rate(sample_rate=32000)
    csc.add_signal_preprocessing(preprocess_dict=dict(mono=True))
    csc.add_extraction_method(MelSpectrogramExtractionMethod(**extraction_params))

    keys = list(csc.get_keys())
    comb_iterator = itertools.chain(*map(lambda x: itertools.combinations(keys, x), range(0, len(keys) + 1)))

    for combo in comb_iterator:
        key_list = list(combo)
        fold = 0
        for train, test in csc.generate_training_splits(key_list):
            for i in range(2):
                if i == 0:
                    model = mtmf.create_model(MultiTaskHardSharing.__name__,
                                              input_size=train[0].shape[-1],
                                              task_list=train.get_task_list())
                else:
                    model = mtmf.create_model(MultiTaskHardSharingConvolutional.__name__,
                                              task_list=train.get_task_list())

                print('Model Created')

                results = Training.create_results(modelname=model.name,
                                                  task_list=train.get_task_list(),
                                                  fold=fold,
                                                  results_path=drive + r":\Thesis_Results",
                                                  num_epochs=meta_params['num_epochs'])

                Training.run_gradient_descent(model=model,
                                              concat_dataset=train,
                                              results=results,
                                              batch_size=meta_params['batch_size'],
                                              num_epochs=meta_params['num_epochs'],
                                              learning_rate=meta_params['learning_rate'],
                                              test_dataset=test)
                results.close_writer()
                fold += 1


# tensorboard --logdir F:\Thesis_Results\Training_Results\experiments
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
