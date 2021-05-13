from __future__ import print_function, division

from Tests.conv_test import *

run_names = ['Result_20_04_2021_18_10_40', 'Result_15_04_2021_16_34_08', 'Result_19_04_2021_05_37_13',
             'Result_08_04_2021_15_33_45', 'Result_08_04_2021_13_15_48', 'Result_17_04_2021_05_27_55',
             'Result_14_04_2021_23_48_13']

drive = 'F'
meta_params = read_config('meta_params_cnn_MelSpectrogram')
model_checkpoints_path = drive + r":\Thesis_Results\Model_Checkpoints"
dataset_list = [[2, 0], [1, 2], [2, 4], [2], [1], [1, 4], [1, 0]]

for d in range(len(dataset_list)):
    print(dataset_list[d])
    training_dataset, eval_dataset = get_concat(dataset_list[d])
    task_list = training_dataset.get_task_list()
    results = Results(run_name=run_names[d], model_checkpoints_path=model_checkpoints_path)
    run_test(eval_dataset=eval_dataset, meta_params=meta_params, run_name=run_names[d],
             model_checkpoints_path=model_checkpoints_path)
    run_name = results.run_name
    del results
    torch.cuda.empty_cache()
