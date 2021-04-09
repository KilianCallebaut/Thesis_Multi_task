from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tests.config_reader import *
from Training.Training import *

model_params = read_config('model_params_cnn')
extraction_params = read_config('extraction_params_cnn_MelSpectrogram')
chenaudio = ChenAudiosetDataset(**extraction_params)
chenaudio.prepare_taskDatasets(**dict(test_size=0.2, window_size=64, hop_size=32, dic_of_labels_limits={"None of the above": 500}))
training = ConcatTaskDataset([chenaudio.toTrainTaskDataset()])
model = MultiTaskHardSharingConvolutional(1, **model_params, task_list=training.get_task_list())

checkpoint = torch.load("E:\Thesis_Results\Model_Checkpoints\Result_02_04_2021_01_13_52epoch_57.pth")

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
mode=model.cuda()

loader = torch.utils.data.DataLoader(training, batch_size=1, shuffle=True)

task_list = training.get_task_list()
task_labels = []
task_predictions = []
perc = 0
for inputs, labels, names in loader:
    print(perc / len(loader), end='\r')
    perc += 1
    inputs = inputs.cuda()
    labels = labels.cuda()
    output = model(inputs)
    task_labels = task_labels + [Training.get_actual_labels(l[None, :], task_list[0]).tolist() for l in labels]
    task_predictions = task_predictions + [Training.get_actual_output(l[None, :], task_list[0]).tolist() for l in
                                           output]

met = metrics.classification_report(task_labels, task_predictions, output_dict=True)
