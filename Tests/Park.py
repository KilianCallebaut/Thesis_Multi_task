import os
import sys

import torch
from torch import optim

from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.ExtractionMethod import NeutralExtractionMethod, PerDimensionScaling
from MultiTask.ParkClassifier import ParkClassifier
from Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from Training.Training import Training

drive = r"E:/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGGISHExtract(NeutralExtractionMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.eval()

    def extract_features(self, sig_samplerate) -> torch.tensor:
        with torch.no_grad():
            return self.model.forward(*sig_samplerate).to(torch.device("cpu"))


def main(argv):
    csc = ConcatTrainingSetCreator(nr_runs=4, random_state=444)
    csc.add_data_reader(ChenAudiosetDataset(mode=2))

    csc.add_sample_rate(sample_rate=16000)
    csc.add_extraction_method(extraction_method=PerDimensionScaling(
        VGGISHExtract(
            name='VGGishPark'
        )
    )
    )
    csc.add_sampling(dict(others=500))

    for i in range(2):
        generator = csc.generate_training_splits()
        train, test, _ = next(generator)
        model = ParkClassifier(train.get_task_list())
        results = Training.create_results(modelname=model._get_name(),
                                          task_list=train.get_task_list(),
                                          results_path=os.path.join(drive, 'Thesis_Results', 'Park'),
                                          num_epochs=200)
        Training.run_gradient_descent(model=model,
                                      concat_dataset=train,
                                      test_dataset=test,
                                      results=results,
                                      batch_size=256,
                                      num_epochs=200,
                                      optimizer=optim.SGD(model.parameters(), lr=0.001))
        csc.add_sampling(dict(others=500, speech=0))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
