import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'
# load/save model_checkpoints
# load/save matrices
# load/save classification report
class Results:
    audioset_train_path = drive+r":\Thesis_Results\Training_Results"
    audioset_eval_path = drive+r":\Thesis_Results\Evaluation_Results"
    audioset_file_base = r"Result"
    model_checkpoints_path = drive+r":\Thesis_Results\Model_Checkpoints"

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

        self.writer = SummaryWriter(log_dir='experiments/'+self.run_name)
        self.num_epochs = kwargs.get('num_epochs')
        self.training_curve = np.zeros(self.num_epochs)
        self.evaluation_curve = np.zeros(self.num_epochs)

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
        joblib.dump(mat, os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}".format(phase, task.name, epoch)))
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
        return joblib.load(os.path.join(path, self.run_name, "{}_conf_mat_{}_epoch_{}".format(phase, task.name, epoch)))

    def add_class_report(self, epoch, report, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        joblib.dump(report, os.path.join(path, self.run_name, "{}_class_report_{}_epoch_{}".format(phase, task.name, epoch)))

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
        return joblib.load(os.path.join(path, self.run_name, "{}_class_report_{}_epoch_{}".format(phase, task.name, epoch)))

    def add_loss_to_curve(self, epoch, step, loss, train):

        if train:
            path = self.audioset_train_path
            phase = 'Train'
            self.training_curve[epoch] = loss
            joblib.dump(self.training_curve, os.path.join(path, self.run_name, "{}_losscurve".format(phase)))
        else:
            path = self.audioset_eval_path
            phase = 'Evaluation'
            self.evaluation_curve[epoch] = loss
            joblib.dump(self.evaluation_curve, os.path.join(path, self.run_name, "{}_losscurve".format(phase)))
        self.writer.add_scalar("{}/Loss".format(phase), loss / step, epoch)

    def load_loss_curve(self, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(os.path.join(path, self.run_name, "{}_losscurve".format(phase)))

    def add_loss_to_curve_task(self, epoch, step, loss, task, train):
        if train:
            path = self.audioset_train_path
            phase = 'Train'
            self.training_curve[epoch] = loss
            joblib.dump(self.training_curve, os.path.join(path, self.run_name, "{}_losscurve_{}".format(phase, task.name)))
        else:
            path = self.audioset_eval_path
            phase = 'Evaluation'
            self.evaluation_curve[epoch] = loss
            joblib.dump(self.evaluation_curve, os.path.join(path, self.run_name, "{}_losscurve_{}".format(phase, task.name)))

        self.writer.add_scalar("{}/Loss/{}".format(phase, task.name), loss / step, epoch)

    def load_loss_curve_task(self, task, train):
        path = self.audioset_eval_path
        phase = 'Evaluation'
        if train:
            path = self.audioset_train_path
            phase = 'Train'
        return joblib.load(os.path.join(path, self.run_name, "{}_losscurve_{}".format(phase, task.name)))

    def flush_writer(self):
        self.writer.flush()

    def close_writer(self):
        self.writer.close()

    @staticmethod
    def create_model_loader(name, **kwargs):
        return Results(run_name=name, **kwargs)
