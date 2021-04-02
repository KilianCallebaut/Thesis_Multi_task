from __future__ import print_function, division

import sys

from Tests.config_reader import *

# from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
# from DataReaders.DCASE2017_SS import DCASE2017_SS
#
# from DataReaders.ExtractionMethod import Mfcc, MelSpectrogram
# from DataReaders.FSDKaggle2018 import FSDKaggle2018
# from DataReaders.Ravdess import Ravdess
# from DataReaders.SpeechCommands import SpeechCommands
# from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Training.Training import Training


def main(argv):
    meta_params = read_config('meta_params_cnn_MelSpectrogram')
    extraction_params = read_config('extraction_params_cnn_MelSpectrogram')

    # asvspoof = ASVspoof2015(**extraction_params)
    # asvspoof.prepare_taskDatasets(**read_config('preparation_params_asvspoof_cnn'))

    chenaudio = ChenAudiosetDataset(**extraction_params)
    chenaudio.prepare_taskDatasets(**read_config('preparation_params_chen_cnn'))

    # dcaseScene = DCASE2017_SS(**extraction_params)
    # dcaseScene.prepare_taskDatasets(**read_config('preparation_params_dcaseScene_cnn'))
    #
    # fsdkaggle = FSDKaggle2018(**extraction_params)
    # fsdkaggle.prepare_taskDatasets(**read_config('preparation_params_fsdkaggle_cnn'))
    #
    # ravdess = Ravdess(**extraction_params)
    # ravdess.prepare_taskDatasets(**read_config('preparation_params_ravdess_cnn'))
    #
    # speechcommands = SpeechCommands(**extraction_params)
    # speechcommands.prepare_taskDatasets(**read_config('preparation_params_speechcommands_cnn'))
    print('loaded all datasets')

    # asvspoof_t = asvspoof.toTrainTaskDataset()
    chen_t = chenaudio.toTrainTaskDataset()
    # dcasScent_t = dcaseScene.toTrainTaskDataset()
    # fsdkaggle_t = fsdkaggle.toTrainTaskDataset()
    # ravdess_t = ravdess.toTrainTaskDataset()
    # speechcommands_t = speechcommands.toTrainTaskDataset()
    taskdatasets = [chen_t]
    # taskdatasets = [asvspoof_t, chen_t, dcasScent_t, fsdkaggle_t, ravdess_t, speechcommands_t]

    # asvspoof_e = asvspoof.toValidTaskDataset()
    # chen_e = chenaudio.toTestTaskDataset()
    # dcasScent_e = dcaseScene.toValidTaskDataset()
    # fsdkaggle_e = fsdkaggle.toTestTaskDataset()
    # ravdess_e = ravdess.toTestTaskDataset()
    # speechcommands_e = speechcommands.toValidTaskDataset()
    # evaldatasets = [chen_e]
    # evaldatasets = [asvspoof_e,  chen_e, dcasScent_e, fsdkaggle_e, ravdess_e, speechcommands_e]

    print('Done loading')

    print('test loop')
    for i in range(len(taskdatasets)):
        print(taskdatasets[i].task.name)
        training_dataset = ConcatTaskDataset([taskdatasets[i]])
        # eval_dataset = ConcatTaskDataset([evaldatasets[i]])
        task_list = training_dataset.get_task_list()
        model = MultiTaskHardSharingConvolutional(1,
                                                  **read_config('model_params_cnn'),
                                                  task_list=task_list)
        model = model.cuda()
        print('Model Created')

        model, results = Training.run_gradient_descent(model=model,
                                                       concat_dataset=training_dataset,
                                                       batch_size=meta_params['batch_size'],
                                                       num_epochs=meta_params['num_epochs'],
                                                       learning_rate=meta_params['learning_rate'])
        # Training.evaluate(blank_model=model,
        #                   concat_dataset=eval_dataset,
        #                   training_results=results,
        #                   batch_size=meta_params['batch_size'],
        #                   num_epochs=meta_params['num_epochs'])

        # for j in range(i + 1, len(taskdatasets)):
        #     print(taskdatasets[i].task.name + ' combined with ' + taskdatasets[j].task.name)
        #     training_dataset = ConcatTaskDataset([taskdatasets[i], taskdatasets[j]])
        #     eval_dataset = ConcatTaskDataset([evaldatasets[i], taskdatasets[j]])
        #     task_list = training_dataset.get_task_list()
        #     model = MultiTaskHardSharing(taskdatasets[i].inputs[0].shape[0], meta_params['nodes_in_layer'],
        #                                  meta_params['amount_shared_layers'], task_list)
        #     model = model.cuda()
        #     print('Model Created')
        #
        #     model, results = Training.run_gradient_descent(model=model,
        #                                                    concat_dataset=training_dataset,
        #                                                    batch_size=meta_params['batch_size'],
        #                                                    num_epochs=meta_params['num_epochs'],
        #                                                    learning_rate=meta_params['learning_rate'])
        #     Training.evaluate(blank_model=model,
        #                       concat_dataset=eval_dataset,
        #                       training_results=results,
        #                       batch_size=meta_params['batch_size'],
        #                       num_epochs=meta_params['num_epochs'])

    return 0


# tensorboard --logdir Tests/runs
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
