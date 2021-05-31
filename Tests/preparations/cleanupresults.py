import os
import shutil

experiments_folder = r'F:\Thesis_Results\Training_Results\experiments'
training_folder = r'F:\Thesis_Results\Training_Results'
model_checkpoints_folder = r'F:\Thesis_Results\Model_Checkpoints'
evaluation_folder = r'F:\Thesis_Results\Evaluation_Results'

exp_list = os.listdir(experiments_folder)
filtered_training = [d for d in os.listdir(training_folder) if d not in exp_list and d != 'experiments']
filtered_model = [d for d in os.listdir(model_checkpoints_folder) if d.split('epoch')[0] not in exp_list]
filtered_eval = [d for d in os.listdir(evaluation_folder) if d not in exp_list]

print('ok')

for t in filtered_training:
    shutil.rmtree(os.path.join(training_folder, t))

for t in filtered_model:
    os.remove(os.path.join(model_checkpoints_folder, t))

for t in filtered_eval:
    if os.path.isfile(os.path.join(evaluation_folder, t)):
        os.remove(os.path.join(evaluation_folder, t))
    else:
        shutil.rmtree(os.path.join(evaluation_folder, t))
