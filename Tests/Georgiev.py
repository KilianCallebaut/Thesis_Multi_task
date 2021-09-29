import itertools
import os
import sys

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.ExtractionMethod import PerCelScaling, LogbankSummaryExtraction, \
    NeutralExtractionMethod
from DataReaders.Ravdess import Ravdess
from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from MultiTask.MultiTaskModelFactory import MultiTaskModelFactory
from Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from Training.Training import Training

drive = r"E:/"


def main(argv):
    csc = ConcatTrainingSetCreator(nr_runs=5)
    csc.add_data_reader(DCASE2017_SS(object_path=drive + r'Thesis_Results\Data_Readers\DCASE2017_SS_{}',
                                     data_path=drive + r'Thesis_Datasets\DCASE2017'))
    csc.add_data_reader(Ravdess(object_path=drive + r'Thesis_Results\Data_Readers\Ravdess',
                                data_path=drive + r'Thesis_Datasets\Ravdess'))
    csc.add_data_reader(Ravdess(object_path=drive + r'Thesis_Results\Data_Readers\Ravdess',
                                data_path=drive + r'Thesis_Datasets\Ravdess',
                                mode=1))
    csc.add_data_reader(ASVspoof2015(object_path=os.path.join(drive, r'Thesis_Results\Data_Readers\ASVspoof2015'),
                                     data_path=drive + r'Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015'))
    csc.add_sample_rate(8000)
    csc.add_extraction_method(extraction_method=LogbankSummaryExtraction(
        PerCelScaling(NeutralExtractionMethod()),
        extraction_params=dict(winlen=0.03, winstep=0.01, nfilt=24, nfft=512)), multiply=False)
    csc.add_transformation_call('normalize_fit')
    csc.add_transformation_call('normalize_inputs')

    mtmf = MultiTaskModelFactory()
    mtmf.add_modelclass(MultiTaskHardSharing)
    mtmf.add_static_model_parameters(MultiTaskHardSharing.__name__, **{"hidden_size": 512, "n_hidden": 4})
    keys = list(csc.get_keys())
    comb_iterator = itertools.chain(*map(lambda x: itertools.combinations(keys, x), range(1, len(keys) + 1)))

    for combo in comb_iterator:
        key_list = list(combo)
        for train, test, fold in csc.generate_training_splits(key_list):
            model = mtmf.create_model(MultiTaskHardSharing.__name__,
                                      input_size=train.datasets[0].get_input(0).flatten().shape[0],
                                      task_list=train.get_task_list())
            results = Training.create_results(modelname=model.name, task_list=train.get_task_list(), fold=fold,
                                              results_path=os.path.join(drive, 'Thesis_Results'), num_epochs=200)
            Training.run_gradient_descent(model=model, concat_dataset=train, test_dataset=test, results=results,
                                          learning_rate=0.001, batch_size=32, num_epochs=200)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
