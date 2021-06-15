import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'


# load/save model_checkpoints
# load/save matrices
# load/save classification report
class Results:
    audioset_train_path = drive + r":\Thesis_Results\Training_Results"
    audioset_eval_path = drive + r":\Thesis_Results\Evaluation_Results"
    audioset_file_base = r"Result"
    model_checkpoints_path = drive + r":\Thesis_Results\Model_Checkpoints"

    def __init__(self, **kwargs):

        if 'run_name' in kwargs:
            self.run_name = kwargs.get('run_name')
        else:
            self.run_name = self.audioset_file_base + "_" + str(
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if 'audioset_train_path' in kwargs:
            self.audioset_train_path = kwargs.get('audioset_train_path')
        if 'audioset_eval_path' in kwargs:
            self.audioset_eval_path = kwargs.get('audioset_eval_path')

        if not os.path.exists(os.path.join(self.audioset_train_path, self.run_name)):
            os.makedirs(os.path.join(self.audioset_train_path, self.run_name))
        if not os.path.exists(os.path.join(self.audioset_eval_path, self.run_name)):
            os.makedirs(os.path.join(self.audioset_eval_path, self.run_name))
        if 'model_checkpoints_path' in kwargs:
            self.model_checkpoints_path = kwargs.pop('model_checkpoints_path')

        self.writer = SummaryWriter(log_dir=os.path.join(self.audioset_train_path, 'experiments', self.run_name))
        self.num_epochs = kwargs.get('num_epochs')
        self.training_curve = np.zeros(self.num_epochs)
        self.evaluation_curve = np.zeros(self.num_epochs)
        self.training_curve_task = {}
        self.evaluation_curve_task = {}

    def add_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        torch.save({'epoch': nr_epoch, 'model_state_dict': model.state_dict()}, path)

    def load_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    def add_confusion_matrix(self, epoch, mat, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        joblib.dump(mat,
                    os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))
        fig = plt.figure()
        plt.imshow(mat)
        self.writer.add_figure('{}/Confusion matrix/{}'.format(phase, task.name), fig, epoch)
        plt.close(fig)
        print(mat)

    def load_confusion_matrix(self, epoch, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(
            os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))

    def add_multi_confusion_matrix(self, epoch, mat, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        joblib.dump(mat,
                    os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))
        print(mat)

    def load_multi_confusion_matrix(self, epoch, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(
            os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}.gz".format(phase, task.name, epoch)))

    def add_class_report(self, epoch, report, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
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
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(
            os.path.join(path, self.run_name, "{}_class_report_{}_epoch_{}.gz".format(phase, task.name, epoch)))

    def add_loss_to_curve(self, epoch, step, loss, train):
        if train:
            phase = 'Train'
            self.training_curve[epoch] = loss /step
        else:
            phase = 'Evaluation'
            self.evaluation_curve[epoch] = loss /step
        self.writer.add_scalar("{}/Loss".format(phase), loss / step, epoch)

    def load_loss_curve(self, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(os.path.join(path, self.run_name, "{}_losscurve.gz".format(phase)))

    def write_loss_curves(self):
        phase = 'Train'
        path = self.audioset_train_path
        joblib.dump(self.training_curve, os.path.join(path, self.run_name, "{}_losscurve.gz".format(phase)))
        phase = 'Evaluation'
        path = self.audioset_eval_path
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
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(os.path.join(path, self.run_name, "{}_losscurve_tasks.gz".format(phase, task.name)))

    def write_loss_curve_tasks(self):
        path = self.audioset_train_path
        phase = 'Train'
        joblib.dump(self.training_curve_task, os.path.join(path, self.run_name, "{}_losscurve_tasks.gz".format(phase)))
        path = self.audioset_eval_path
        phase = 'Evaluation'
        joblib.dump(self.evaluation_curve_task,
                    os.path.join(path, self.run_name, "{}_losscurve_tasks.gz".format(phase)))

    def early_stop(self, epoch):
        if epoch == len(self.training_curve) - 1:
            return True
        if epoch == 0:
            return False

        if (self.training_curve[epoch-1] - self.training_curve[epoch]) < 0.1:
            return True

        return False

    def flush_writer(self):
        self.writer.flush()

    def close_writer(self):
        self.writer.close()

    @staticmethod
    def create_model_loader(name, **kwargs):
        return Results(run_name=name, **kwargs)
