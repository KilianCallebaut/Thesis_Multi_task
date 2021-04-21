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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load/save model_checkpoints
# load/save matrices
# load/save classification report
class Results:
    audioset_train_path = r"D:\Thesis_Results\Training_Results"
    audioset_eval_path = r"D:\Thesis_Results\Evaluation_Results"
    audioset_file_base = r"Result"
    model_checkpoints_path = r"F:\Thesis_Results\Model_Checkpoints"

    def __init__(self, **kwargs):

        if 'run_name' in kwargs:
            self.run_name = kwargs.get('run_name')
        else:
            self.run_name = self.audioset_file_base + "_" + str(
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if not os.path.exists(os.path.join(self.audioset_train_path, self.run_name)):
            os.makedirs(os.path.join(self.audioset_train_path, self.run_name))
        if not os.path.exists(os.path.join(self.audioset_eval_path, self.run_name)):
            os.makedirs(os.path.join(self.audioset_eval_path, self.run_name))
        if 'model_checkpoints_path' in kwargs:
            self.model_checkpoints_path = kwargs.pop('model_checkpoints_path')

    def add_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name, "epoch_{}.pth".format(nr_epoch))
        torch.save({'epoch': nr_epoch, 'model_state_dict': model.state_dict()}, path)

    def load_model_parameters(self, nr_epoch, model):
        path = os.path.join(self.model_checkpoints_path, self.run_name + "epoch_{}.pth".format(nr_epoch))
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    def add_matrix(self, epoch, mat, train):
        path = self.audioset_eval_path
        if train:
            path = self.audioset_train_path
        joblib.dump(mat, os.path.join(path, self.run_name, "conf_mat_epoch_{}".format(epoch)))

    def load_matrix(self, epoch, train):
        path = self.audioset_eval_path
        if train:
            path = self.audioset_train_path
        return joblib.load(os.path.join(path, self.run_name, "conf_mat_epoch_{}".format(epoch)))

    def add_class_report(self, epoch, report, train):
        path = self.audioset_eval_path
        if train:
            path = self.audioset_train_path
        joblib.dump(report, os.path.join(path, self.run_name, "class_report_epoch_{}".format(epoch)))

    def load_class_report(self, epoch, train):
        path = self.audioset_eval_path
        if train:
            path = self.audioset_train_path
        return joblib.load(os.path.join(path, self.run_name, "class_report_epoch_{}".format(epoch)))

    @staticmethod
    def create_model_loader(name, **kwargs):
        return Results(run_name=name, **kwargs)
