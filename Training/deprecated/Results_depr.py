import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import torch
from sklearn import metrics

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


class Results:
    audioset_train_path = r"E:\Thesis_Results\Training_Results"
    audioset_eval_path = r"E:\Thesis_Results\Evaluation_Results"
    audioset_file_base = r"Result"
    model_checkpoints_path = r"F:\Thesis_Results\Model_Checkpoints"

    # def __init__(self, concat_dataset: ConcatTaskDataset, nr_epochs: int, *args, **kwargs):
    def __init__(self, **kwargs):

        # epochs, steps, (task -> ([outputs], [targets], loss), total loss)
        self.all_results = [list() for _ in range(kwargs.get('nr_epochs'))]

        # (nr_epochs, nr_batches_in_epoch, nr_tasks)
        self.concat_dataset = kwargs.get('concat_dataset')
        self.batch_size = kwargs.get('batch_size')
        self.learning_rate = kwargs.get('learning_rate')
        self.weight_decay = kwargs.get('weight_decay')
        self.nr_epochs = kwargs.get('nr_epochs')
        if 'run_name' in kwargs:
            self.run_name = kwargs.get('run_name')
        else:
            self.run_name = self.audioset_file_base + "_" + str(
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if 'model_checkpoints_path' in kwargs:
            self.model_checkpoints_path = kwargs.pop('model_checkpoints_path')

    # epochs, steps, (total_loss_batch, tasks -> [([outputs], [targets], loss))]
    def add_output(self, nr_epoch, output_batch, labels_batch, losses_batch, total_loss_batch):
        batch = list()
        for i in range(len(output_batch)):
            batch.append((output_batch[i], labels_batch[i], losses_batch[i]))
        self.all_results[nr_epoch].append((total_loss_batch, batch))

    def add_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        torch.save({'epoch': nr_epoch, 'model_state_dict': model.state_dict()}, path)

    def load_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])

    def flatten_epoch_output_target_per_task(self, nr_epoch):
        return [[(step[1][task_idx][0][idx].tolist(),
                  step[1][task_idx][1][idx].tolist())  # pred, target
                 for step in self.all_results[nr_epoch]
                 for idx in range(len(step[1][task_idx][0]))]
                for task_idx in range(len(self.concat_dataset.get_task_list()))]

    def calculate_true_predicted_epoch_per_task(self, nr_epoch):
        pred_target_per_task = self.flatten_epoch_output_target_per_task(nr_epoch)
        return [[self.get_actual_labels(output=torch.Tensor([pred_tar[0]]), target=torch.Tensor([pred_tar[1]]),
                                        task=self.concat_dataset.get_task_list()[task_idx])
                 for pred_tar in pred_target_per_task[task_idx]] for task_idx in range(len(pred_target_per_task))]

    # tasks, dict
    def calculate_metrics_epoch_per_task(self, nr_epoch):
        pred_target_per_task = self.calculate_true_predicted_epoch_per_task(nr_epoch)
        return [metrics.classification_report([pred_tar[1].tolist() for pred_tar in pred_target_per_task[task_idx]],
                                              [pred_tar[0].tolist() for pred_tar in pred_target_per_task[task_idx]],
                                              target_names=self.concat_dataset.get_task_list()[task_idx].output_labels,
                                              output_dict=True)
                for task_idx in range(len(pred_target_per_task))]

    def calculate_logloss_epoch_per_task(self, nr_epoch):
        pred_target_per_task = self.flatten_epoch_output_target_per_task(nr_epoch)
        return [metrics.log_loss([pred_tar[1] for pred_tar in pred_target_per_task[task_idx]],
                                 [pred_tar[0] for pred_tar in pred_target_per_task[task_idx]])
                for task_idx in range(len(pred_target_per_task))]

    def read_files(self, name, train):
        audioset_path = self.audioset_eval_path
        if train:
            audioset_path = self.audioset_train_path
        # info = cPickle.load(open(audioset_path + "/" + name, 'rb'))
        info = joblib.load(audioset_path + "/" + name)
        self.all_results = info['all_results']
        self.concat_dataset = info['concat_dataset']
        self.batch_size = info['batch_size']
        self.learning_rate = info['learning_rate']
        self.weight_decay = info['weight_decay']
        self.nr_epochs = info['nr_epochs']

    def write_files(self, train):
        dict = {'all_results': self.all_results,
                # 'concat_dataset': self.concat_dataset,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'nr_epochs': self.nr_epochs}
        # cPickle.dump(dict, open(self.get_write_file_path(train), 'wb'))
        joblib.dump(dict, self.get_write_file_path(train))

    def get_actual_labels(self, output, target, task):
        if task.output_module == "softmax":
            translated_out = torch.max(output, 1)[1]
            translated_target = torch.max(target, 1)[1]
            return translated_out, translated_target
        elif task.output_module == "sigmoid":
            translated_out = (output >= 0.5).float()
            return torch.squeeze(translated_out), torch.squeeze(target)

    def get_write_file_path(self, train):
        audioset_path = self.audioset_eval_path
        if train:
            audioset_path = self.audioset_train_path

        return os.path.join(audioset_path, self.run_name + ".p")

    def plot_training_curve_loss_overall(self):
        logloss_overall = [self.calculate_logloss_epoch_per_task(i) for i in range(self.nr_epochs)]

        # plotting
        plt.title("Training Curve Loss (batch_size={}, lr={})".format(self.batch_size, self.learning_rate))
        task_list = self.concat_dataset.get_task_list()
        for t_id in range(len(task_list)):
            plt.plot([i for i in range(self.nr_epochs)], [l[t_id] for l in logloss_overall], label=task_list[t_id])
        plt.xlabel("Iterations")
        plt.ylabel("Overall Loss")
        plt.legend()
        plt.show()

    def plot_evaluation_curve_on_dataset(self):
        logloss_overall = [self.calculate_logloss_epoch_per_task(i) for i in range(self.nr_epochs)]

        # plotting
        plt.title("Evaluation Curve Loss (batch_size={}, lr={})".format(self.batch_size, self.learning_rate))
        task_list = self.concat_dataset.get_task_list()
        for t_id in range(len(task_list)):
            plt.plot([i for i in range(self.nr_epochs)], [l[t_id] for l in logloss_overall], label=task_list[t_id])
        plt.xlabel("Iterations")
        plt.ylabel("Overall Loss")
        plt.legend()
        plt.show()

    @staticmethod
    def create_from_file(name):
        # info = cPickle.load(open(os.path.join(r"E:\Thesis_Results", name), 'rb'))
        info = joblib.load(os.path.join(r"E:\Thesis_Results", name), 'rb')
        all_results = info['all_results']
        concat_dataset = info['concat_dataset']
        batch_size = info['batch_size']
        learning_rate = info['learning_rate']
        weight_decay = info['weight_decay']
        nr_epochs = info['nr_epochs']
        r = Results(concat_dataset, batch_size, learning_rate, weight_decay, nr_epochs)
        r.all_results = all_results
        return r

    @staticmethod
    def create_model_loader(name):
        return Results(run_name=name)
