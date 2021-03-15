from __future__ import print_function, division

import sys

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.DCASE2017_SS_Eval import DCASE2017_SS_Eval
from DataReaders.FSDKaggle2018 import FSDKaggle2018
from DataReaders.Ravdess import Ravdess
from DataReaders.SpeechCommands import SpeechCommands
from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Training.Training import Training


def main(argv):
    meta_params = {
        'extraction_method': 'logmel',
        'batch_size': 8,
        'num_epochs': 500,
        'learning_rate': 0.001
    }

    data_reader_params = dict(
        extraction_method='logmel',

        nfft=1024,
        hopsize=320,
        mel_bins=96,
        window='hann',
        lowfreq=50
    )

    model_params = dict(
        hidden_size=64,
        n_hidden=4
    )

    dcaseScene = DCASE2017_SS(**data_reader_params)
    asvspoof = ASVspoof2015(**data_reader_params)

    chenaudio = ChenAudiosetDataset(**data_reader_params)
    speechcommands = SpeechCommands(**data_reader_params)
    ravdess = Ravdess(**data_reader_params)
    fsdkaggle = FSDKaggle2018(**data_reader_params)
    #
    dcasScent_t = dcaseScene.toTrainTaskDataset()
    asvspoof_t = asvspoof.toTrainTaskDataset()
    chen_t = chenaudio.toTrainTaskDataset()
    speechcommands_t = speechcommands.toTrainTaskDataset()
    ravdess_t = ravdess.toTrainTaskDataset()
    fsdkaggle_t = fsdkaggle.toTrainTaskDataset()
    # taskdatasets = [dcasScent_t]
    taskdatasets = [dcasScent_t, asvspoof_t, chen_t, speechcommands_t, ravdess_t, fsdkaggle_t]

    dcasScent_e = dcaseScene.toTestTaskDataset()
    asvspoof_e = asvspoof.toTestTaskDataset()
    chen_e = chenaudio.toTestTaskDataset()
    speechcommands_e = speechcommands.toTestTaskDataset()
    ravdess_e = ravdess.toTestTaskDataset()
    fsdkaggle_e = fsdkaggle.toTestTaskDataset()
    # evaldatasets = [dcasScent_e]
    evaldatasets = [dcasScent_e, asvspoof_e, chen_e, speechcommands_e, ravdess_e, fsdkaggle_e]

    print('Done loading')

    print('test loop')
    for i in range(len(taskdatasets)):
        print(taskdatasets[i].task.name)
        training_dataset = ConcatTaskDataset([taskdatasets[i]])
        eval_dataset = ConcatTaskDataset([evaldatasets[i]])
        task_list = training_dataset.get_task_list()
        model = MultiTaskHardSharingConvolutional(taskdatasets[i].inputs[0].shape[0],
                                                  **model_params,
                                                  task_list=task_list)
        model = model.cuda()
        print('Model Created')

        model, results = Training.run_gradient_descent(model=model,
                                                       concat_dataset=training_dataset,
                                                       batch_size=meta_params['batch_size'],
                                                       num_epochs=meta_params['num_epochs'],
                                                       learning_rate=meta_params['learning_rate'])
        Training.evaluate(blank_model=model,
                          concat_dataset=eval_dataset,
                          training_results=results,
                          batch_size=meta_params['batch_size'],
                          num_epochs=meta_params['num_epochs'])

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
