import os
from datetime import datetime
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

from Tasks.Task import Task

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'


# load/save model_checkpoints
# load/save matrices
# load/save classification report
class Results:
    training_results_path = drive + r":\Thesis_Results\Training_Results"
    evaluation_results_path = drive + r":\Thesis_Results\Evaluation_Results"
    audioset_file_base = r"Result"
    model_checkpoints_path = drive + r":\Thesis_Results\Model_Checkpoints"

    def __init__(self,
                 num_epochs: int,
                 task_list: List[Task] = None,
                 run_name=None,
                 training_results_path=None,
                 evaluation_results_path=None,
                 model_checkpoints_path=None,
                 tensorboard_folder=None):

        if run_name:
            self.run_name = run_name
        else:
            self.run_name = self.audioset_file_base + "_" + str(
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if training_results_path:
            self.training_results_path = training_results_path
        if evaluation_results_path:
            self.evaluation_results_path = evaluation_results_path
        if not tensorboard_folder:
            self.tensorboard_folder = 'TensorBoard'
        else:
            self.tensorboard_folder = tensorboard_folder

        if not os.path.exists(os.path.join(self.training_results_path, self.run_name)):
            os.makedirs(os.path.join(self.training_results_path, self.run_name))
        if not os.path.exists(os.path.join(self.evaluation_results_path, self.run_name)):
            os.makedirs(os.path.join(self.evaluation_results_path, self.run_name))
        if not os.path.exists(os.path.join(self.training_results_path, self.tensorboard_folder)):
            os.makedirs(os.path.join(self.training_results_path, self.tensorboard_folder))

        if model_checkpoints_path:
            self.model_checkpoints_path = model_checkpoints_path

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.training_results_path, self.tensorboard_folder, self.run_name))
        self.num_epochs = num_epochs
        self.training_curve = np.zeros(self.num_epochs)
        self.evaluation_curve = np.zeros(self.num_epochs)
        self.training_curve_task = {}
        self.evaluation_curve_task = {}

        # Epoch Metrics
        self.running_loss = 0
        self.task_list = task_list
        self.task_truths = {}
        self.task_predictions = {}
        self.task_running_losses = {}
        self.reset_epoch_metrics()

    # (n_tasks, n_samples_batch, n_labels) => (n_tasks, n_samples_epoch, n_labels)
    def add_batch_results(self, batch_flags: List[List[bool]], truths_batch: List[torch.tensor],
                          predictions_batch: List[torch.tensor], losses_batch: List[torch.tensor],
                          combined_loss: torch.tensor):
        self.running_loss += combined_loss.item() * len(batch_flags[0])
        for tsk in self.task_list:
            self.task_truths[tsk.name] += truths_batch[self.task_list.index(tsk)].tolist()
            self.task_predictions[tsk.name] += predictions_batch[self.task_list.index(tsk)].tolist()
            self.task_running_losses[tsk.name] += losses_batch[self.task_list.index(tsk)].item() * sum(
                batch_flags[self.task_list.index(tsk)])

    def add_epoch_metrics(self, epoch: int, step: int, training: bool):
        self.add_loss_to_curve(epoch, step, self.running_loss, training)
        for tsk in self.task_list:
            print('TASK {}: '.format(tsk.name), end='')
            print(tsk.output_labels)
            epoch_metrics = metrics.classification_report(self.task_truths[tsk.name], self.task_predictions[tsk.name],
                                                          output_dict=True)
            self.add_class_report(epoch, epoch_metrics, tsk, training)
            self.add_loss_to_curve_task(epoch, step, self.task_running_losses[tsk.name], tsk, training)
            if tsk.classification_type == "multi-class":
                mat = metrics.confusion_matrix(self.task_truths[tsk.name], self.task_predictions[tsk.name])
                self.add_confusion_matrix(epoch, mat, tsk, training)
            elif tsk.classification_type == "multi-label":
                mat = metrics.multilabel_confusion_matrix(self.task_truths[tsk.name], self.task_predictions[tsk.name])
                self.add_multi_confusion_matrix(epoch, mat, tsk, training)
        self.reset_epoch_metrics()

    def reset_epoch_metrics(self):
        self.running_loss = 0
        for t in self.task_list:
            self.task_truths[t.name] = []
            self.task_predictions[t.name] = []
            self.task_running_losses[t.name] = 0

    def add_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        torch.save({'epoch': nr_epoch, 'model_state_dict': model.state_dict()}, path)

    def load_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    def add_confusion_matrix(self, epoch, mat, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        joblib.dump(mat,
                    os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))
        fig = plt.figure()
        plt.imshow(mat)
        self.writer.add_figure('{}/Confusion matrix/{}'.format(phase, task.name), fig, epoch)
        plt.close(fig)
        print(mat)

    def load_confusion_matrix(self, epoch, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        return joblib.load(
            os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))

    def add_multi_confusion_matrix(self, epoch, mat, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        joblib.dump(mat,
                    os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))
        print(mat)

    def load_multi_confusion_matrix(self, epoch, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        return joblib.load(
            os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))

    def add_class_report(self, epoch, report, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        joblib.dump(report,
                    os.path.join(path, self.run_name, "{}_class_report_{}_epoch_{}.gz".format(phase, task.name, epoch)))

        for key, value in report.items():
            if type(value) is dict:
                for key2, value2 in value.items():
                    self.writer.add_scalar("{}/{}/{}/{}".format(phase, key, key2, task.name), value2, epoch)
            else:
                self.writer.add_scalar("{}/{}/{}".format(phase, key, task.name), value, epoch)
        print(report)

    def load_class_report(self, epoch, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        return joblib.load(
            os.path.join(path, self.run_name, "{}_class_report_{}_epoch_{}.gz".format(phase, task.name, epoch)))

    def add_loss_to_curve(self, epoch, step, loss, train):
        if train:
            phase = 'Train'
            self.training_curve[epoch] = loss / step
        else:
            phase = 'Evaluation'
            self.evaluation_curve[epoch] = loss / step
        self.writer.add_scalar("{}/Loss".format(phase), loss / step, epoch)

    def load_loss_curve(self, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        return joblib.load(os.path.join(path, self.run_name, "{}_losscurve.gz".format(phase)))

    def write_loss_curves(self):
        phase = 'Train'
        path = self.training_results_path
        joblib.dump(self.training_curve, os.path.join(path, self.run_name, "{}_losscurve.gz".format(phase)))
        phase = 'Evaluation'
        path = self.evaluation_results_path
        joblib.dump(self.evaluation_curve, os.path.join(path, self.run_name, "{}_losscurve.gz".format(phase)))

    def add_loss_to_curve_task(self, epoch, step, loss, task, train):
        if train:
            phase = 'Train'
            if task.name not in self.training_curve_task:
                self.training_curve_task[task.name] = np.zeros(self.num_epochs)
            self.training_curve_task[task.name][epoch] = loss / step
        else:
            phase = 'Evaluation'
            if task.name not in self.evaluation_curve_task:
                self.evaluation_curve_task[task.name] = np.zeros(self.num_epochs)
            self.evaluation_curve_task[task.name][epoch] = loss / step

        self.writer.add_scalar("{}/Loss/{}".format(phase, task.name), loss / step, epoch)

    def load_loss_curve_task(self, task, train):
        path = self.evaluation_results_path
        phase = 'Evaluation'
        if train:
            path = self.training_results_path
            phase = 'Train'
        return joblib.load(os.path.join(path, self.run_name, "{}_losscurve_tasks.gz".format(phase, task.name)))

    def write_loss_curve_tasks(self):
        path = self.training_results_path
        phase = 'Train'
        joblib.dump(self.training_curve_task, os.path.join(path, self.run_name, "{}_losscurve_tasks.gz".format(phase)))
        path = self.evaluation_results_path
        phase = 'Evaluation'
        joblib.dump(self.evaluation_curve_task,
                    os.path.join(path, self.run_name, "{}_losscurve_tasks.gz".format(phase)))

    def flush_writer(self):
        self.writer.flush()

    def close_writer(self):
        self.writer.close()

    @staticmethod
    def create_model_loader(name, **kwargs):
        return Results(run_name=name, **kwargs)
