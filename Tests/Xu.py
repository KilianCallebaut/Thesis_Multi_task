import itertools
import os
import sys

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.DCASE2017_SE import DCASE2017_SE
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.ExtractionMethod import PerCelScaling, LogbankSummaryExtraction, \
    NeutralExtractionMethod, MelSpectrogramExtraction, PerDimensionScaling, FramePreparation, \
    MinWindowSizePreparationFitter
from DataReaders.Ravdess import Ravdess
from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from Training.Training import Training

drive = r"E:/"


def main(argv):
    csc = ConcatTrainingSetCreator(nr_runs=4)
    csc.add_data_reader(DCASE2017_SS(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SS_{}',
                                     data_path=drive + r'Thesis_Datasets\DCASE2017'))
    csc.add_data_reader(DCASE2017_SE(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SE_{}',
                                     data_path=drive + r'Thesis_Datasets\DCASE2017'))

    csc.add_extraction_method(extraction_method=
        MelSpectrogramExtraction(
            PerDimensionScaling(
                FramePreparation(
                    MinWindowSizePreparationFitter(
                        NeutralExtractionMethod(

                        )
                    )
                )
            ),
            name='MelSpectrogramXu',
            extraction_params=
            dict(
                n_fft=2048,
                hop_length=512
            )
        )
    )

    mtmf = MultiTaskModelFactory()
    mtmf.add_modelclass(MultiTaskHardSharingConvolutional)
    mtmf.add_static_model_parameters(MultiTaskHardSharingConvolutional.__name__, **{"hidden_size": 512, "n_hidden": 4})
    generator = csc.generate_training_splits()
    train, test, _ = next(generator)
    model = mtmf.create_model(MultiTaskHardSharingConvolutional.__name__,
                              input_size=train.datasets[0].get_input(0).flatten().shape[0],
                              task_list=train.get_task_list())
    results = Training.create_results(modelname=model.name,
                                      task_list=train.get_task_list(),
                                      results_path=os.path.join(drive, 'Thesis_Results'),
                                      num_epochs=200)
    Training.run_gradient_descent(model=model,
                                  concat_dataset=train,
                                  test_dataset=test,
                                  results=results,
                                  learning_rate=0.001,
                                  batch_size=32,
                                  num_epochs=200)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
