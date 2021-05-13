import datetime
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from sklearn import metrics
from torch import nn
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Training.Results import Results

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Training:

    # Source: https://www.cs.toronto.edu/~lczhang/321/tut/tut04.pdf
    # Also helpful: https://github.com/sugi-chan/pytorch_multitask/blob/master/pytorch%20multi-task-Copy2.ipynb
    @staticmethod
    def run_gradient_descent(model: nn.Module,
                             concat_dataset: ConcatTaskDataset,
                             results: Results,
                             batch_size=64,
                             learning_rate=0.05,
                             weight_decay=0,
                             num_epochs=50,
                             start_epoch=0,
                             **kwargs):
        datasets = concat_dataset.datasets
        task_list = concat_dataset.get_task_list()
        n_tasks = len(task_list)

        # name = model.name
        # for n in task_list:
        #     name += "_" + n.name
        # writer = SummaryWriter(comment=name)
        # first_shape = np.array(datasets[0].__getitem__(0)[0].shape)
        # writer.add_graph(model, torch.rand(first_shape[None, :, :]).to(device))

        criteria = [nn.BCELoss().to(device) if t.output_module == 'sigmoid' else nn.CrossEntropyLoss().to(device)
                    for t in task_list]
        target_flags = [
            [False for _ in x.pad_before] + [True for _ in x.targets[0]] + [False for _ in x.pad_after]
            for x in datasets]
        train_loader = torch.utils.data.DataLoader(
            concat_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model.train()  # Set model to training mode

        # Epoch
        for epoch in range(start_epoch, num_epochs):

            print('Epoch {}'.format(epoch))
            print('===========================================================')

            running_loss = 0.0
            step = 0
            task_predictions = [[] for _ in task_list]
            task_labels = [[] for _ in task_list]
            task_running_losses = [0 for _ in task_list]

            # before = list(model.parameters())[0].clone()
            perc = 0
            begin = datetime.datetime.now()
            ex_mx = datetime.timedelta(0)
            ex_t = datetime.timedelta(0)
            # iterate over data
            for inputs, labels, names in train_loader:

                # if len(labels) != batch_size:
                #     continue

                # tensors for filtering instances in batch and targets that are not from the task
                # batch_flags = [[True if t_id == n else False for n in names] for t_id in
                #                range(len(task_list))]
                batch_flags = [[True if t.name == n else False for n in names] for t in
                               task_list]

                losses_batch = [torch.tensor([0]).to(device) for _ in task_list]
                output_batch = [torch.Tensor([]) for _ in task_list]
                labels_batch = [torch.Tensor([]) for _ in task_list]

                # define .to(device) on dataloader(s) to make it run on gpu
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizer.zero_grad() to zero parameter gradients
                optimizer.zero_grad()

                # model
                output = model(inputs)

                if perc < (step / len(train_loader)) * 100:
                    perc += 1
                    perc_s = 'I' * perc
                    perc_sp = ' ' * (100 - perc)
                    ex = datetime.datetime.now() - begin
                    begin = datetime.datetime.now()
                    ex_mx = ex if ex > ex_mx else ex_mx
                    ex_t += ex
                    print('[{}{}], execution time: {}, max time: {}, total time: {}'.format(perc_s, perc_sp, ex, ex_mx,
                                                                                            ex_t), end='\r')

                for i in range(n_tasks):
                    if sum(batch_flags[i]) == 0:
                        continue

                    filtered_output = output[i]
                    filtered_labels = labels

                    if n_tasks > 1:
                        filtered_output = filtered_output[batch_flags[i], :]

                        filtered_labels = filtered_labels[batch_flags[i], :]
                        filtered_labels = filtered_labels[:, target_flags[i]]

                    losses_batch[i] = criteria[i](filtered_output,
                                                  Training.calculate_labels(task_list[i].output_module,
                                                                            filtered_labels))
                    output_batch[i] = filtered_output.detach()
                    labels_batch[i] = filtered_labels.detach()

                # training step
                loss = sum(losses_batch)

                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                step += 1
                task_labels = [
                    task_labels[t] +
                    [Training.get_actual_labels(l[None, :], task_list[t]).tolist() for l in labels_batch[t]] for
                    t in range(len(labels_batch))]
                task_predictions = [
                    task_predictions[t] +
                    [Training.get_actual_output(l[None, :], task_list[t]).tolist() for l in output_batch[t]] for
                    t in range(len(output_batch))]
                task_running_losses = [task_running_losses[t] + losses_batch[t].item()
                                       for t in range(len(losses_batch))]

                torch.cuda.empty_cache()

            # Statistics
            # writer.add_scalar("Loss/train", running_loss / step, epoch)
            results.add_loss_to_curve(epoch, step, running_loss, True)
            epoch_metrics = [metrics.classification_report(task_labels[t], task_predictions[t], output_dict=True) for t
                             in range(len(task_predictions))]

            mats = []
            for t in range(n_tasks):
                task_name = task_list[t].name
                print('TASK {}: '.format(task_name), end='')
                results.add_class_report(epoch, epoch_metrics[t], task_list[t], True)
                results.add_loss_to_curve_task(epoch, step, task_running_losses[t], task_list[t], True)

                mat = []
                if task_list[t].output_module == "softmax":
                    # writer.add_scalar("Accuracy/{}".format(task_name), epoch_metrics[t]['accuracy'], epoch)
                    mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                    results.add_confusion_matrix(epoch, mat, task_list[t], True)
                    # fig = plt.figure()
                    # plt.imshow(mat)
                    # writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
                    #                   fig, epoch)
                    # print('Accuracy {} '.format(epoch_metrics[t]['accuracy']), end='')
                elif task_list[t].output_module == "sigmoid":
                    # writer.add_scalar("Micro AVG F1/{}".format(task_name),
                    #                   epoch_metrics[t]['micro avg']['f1-score'], epoch)
                    mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                    # fig = plot_multilabel_confusion(mat, task_list[t].output_labels)
                    # writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
                    #                   fig, epoch)
                    # print('Micro avg F1 {} '.format(epoch_metrics[t]['micro avg']['f1-score']), end='')

                # writer.add_scalar("Macro Avg Precision/{}".format(task_name),
                #                   epoch_metrics[t]['macro avg']['precision'], epoch)
                # print('Macro avg Precision {} '.format(epoch_metrics[t]['macro avg']['precision']), end='')
                # writer.add_scalar("Macro Avg F1/{}".format(task_name), epoch_metrics[t]['macro avg']['f1-score'],
                #                   epoch)
                # print('Macro avg F1 {}'.format(epoch_metrics[t]['macro avg']['f1-score']), end='')
                # writer.add_scalar("Running loss task/{}".format(task_name), task_running_losses[t] / step, epoch)
                # print('Running loss {}'.format(task_running_losses[t] / step))
                print(task_list[t].output_labels)
                print(mat)
                mats.append(mat)

            results.add_model_parameters(epoch, model)

        print('Training Done')

        # for t in range(len(task_predictions)):
        #     mat = []
        #     print(task_list[t].output_labels)
        #     if task_list[t].output_module == "softmax":
        #         mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
        #         fig = plt.figure()
        #         plt.imshow(mat)
        #         writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
        #                           fig, epoch)
        #     elif task_list[t].output_module == "sigmoid":
        #         mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
        #         fig = plot_multilabel_confusion(mat, task_list[t].output_labels)
        #         writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
        #                           fig, epoch)
        #     print(mat)

        # writer.flush()
        # writer.close()
        results.flush_writer()
        print('Wrote Training Results')

        return model, results

    @staticmethod
    def evaluate(blank_model: nn.Module,
                 concat_dataset: ConcatTaskDataset,
                 training_results: Results,
                 batch_size=64,
                 num_epochs=50,
                 start_epoch=0
                 ):

        datasets = concat_dataset.datasets
        task_list = [x.task for x in datasets]
        n_tasks = len(task_list)

        # name = blank_model.name
        # for n in task_list:
        #     name += "_" + n.name
        # writer = SummaryWriter(comment=name + '_evaluation')

        criteria = [nn.BCELoss().to(device) if d.task.output_module == 'sigmoid' else nn.CrossEntropyLoss().to(device)
                    for d in datasets]
        target_flags = [
            [False for _ in x.pad_before] + [True for _ in x.targets[0]] + [False for _ in x.pad_after]
            for x in datasets]
        eval_loader = torch.utils.data.DataLoader(
            concat_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        blank_model.eval()

        with torch.no_grad():
            print("Start Evaluation")
            for epoch in range(start_epoch, num_epochs):
                print('Epoch {}'.format(epoch))
                print('===========================================================')

                training_results.load_model_parameters(epoch, blank_model)

                running_loss = 0.0
                step = 0
                task_predictions = [[] for _ in task_list]
                task_labels = [[] for _ in task_list]
                task_running_losses = [0 for _ in task_list]

                perc = 0
                begin = datetime.datetime.now()
                ex_mx = datetime.timedelta(0)
                ex_t = datetime.timedelta(0)

                for inputs, labels, names in eval_loader:

                    # if len(labels) != batch_size:
                    #     continue

                    # tensors for filtering instances in batch and targets that are not from the task
                    # batch_flags = [[True if t_id == n else False for n in names] for t_id in
                    #                range(len(task_list))]
                    batch_flags = [[True if t.name == n else False for n in names] for t in
                                   task_list]

                    losses_batch = [torch.tensor([0]).to(device) for _ in task_list]
                    output_batch = [torch.Tensor([]) for _ in task_list]
                    labels_batch = [torch.Tensor([]) for _ in task_list]

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output = blank_model(inputs)

                    if perc < (step / len(eval_loader)) * 100:
                        perc += 1
                        perc_s = 'I' * perc
                        perc_sp = ' ' * (100 - perc)
                        ex = datetime.datetime.now() - begin
                        begin = datetime.datetime.now()
                        ex_mx = ex if ex > ex_mx else ex_mx
                        ex_t += ex
                        print(
                            '[{}{}], execution time: {}, max time: {}, total time: {}'.format(perc_s, perc_sp, ex,
                                                                                              ex_mx,
                                                                                              ex_t), end='\r')

                    for i in range(n_tasks):
                        if sum(batch_flags[i]) == 0:
                            continue

                        filtered_output = output[i]
                        filtered_labels = labels

                        if n_tasks > 1:
                            filtered_output = filtered_output[batch_flags[i], :]

                            filtered_labels = filtered_labels[batch_flags[i], :]
                            filtered_labels = filtered_labels[:, target_flags[i]]

                        losses_batch[i] = criteria[i](filtered_output,
                                                      Training.calculate_labels(task_list[i].output_module,
                                                                                filtered_labels))
                        output_batch[i] = filtered_output.detach()
                        labels_batch[i] = filtered_labels.detach()

                    loss = sum(losses_batch)

                    running_loss += loss.item() * inputs.size(0)
                    step += 1
                    task_labels = [
                        task_labels[t] +
                        [Training.get_actual_labels(l[None, :], task_list[t]).tolist() for l in labels_batch[t]] for
                        t in range(len(labels_batch))]
                    task_predictions = [
                        task_predictions[t] +
                        [Training.get_actual_output(l[None, :], task_list[t]).tolist() for l in output_batch[t]] for
                        t in range(len(output_batch))]
                    task_running_losses = [task_running_losses[t] + losses_batch[t].item()
                                           for t in range(len(losses_batch))]

                    torch.cuda.empty_cache()

                # Statistics

                # writer.add_scalar("Loss/eval", running_loss / step, epoch)
                training_results.add_loss_to_curve(epoch, step, running_loss, False)
                epoch_metrics = [metrics.classification_report(task_labels[t], task_predictions[t], output_dict=True)
                                 for t in range(len(task_predictions))]

                mats = []
                for t in range(n_tasks):
                    task_name = task_list[t].name
                    print('TASK {}: '.format(task_name), end='')
                    training_results.add_class_report(epoch, epoch_metrics[t], task_list[t], False)
                    training_results.add_loss_to_curve_task(epoch, step, task_running_losses[t], task_list[t], False)

                    mat = []

                    if task_list[t].output_module == "softmax":
                        # writer.add_scalar("Accuracy/Eval {}".format(task_name), epoch_metrics[t]['accuracy'], epoch)
                        mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                        training_results.add_confusion_matrix(epoch, mat, task_list[t], False)
                        # fig = plt.figure()
                        # plt.imshow(mat)
                        # writer.add_figure('Confusion matrix/Eval {}'.format(task_list[t]),
                        #                   fig, epoch)
                    elif task_list[t].output_module == "sigmoid":
                        # writer.add_scalar("Micro AVG F1/Eval {}".format(task_name),
                        #                   epoch_metrics[t]['micro avg']['f1-score'], epoch)
                        mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                        # fig = plot_multilabel_confusion(mat, task_list[t].output_labels)
                        # writer.add_figure('Confusion matrix/Eval {}'.format(task_list[t]),
                        #                   fig, epoch)

                    # writer.add_scalar("Macro Avg Precision/Eval {}".format(task_name),
                    #                   epoch_metrics[t]['macro avg']['precision'], epoch)
                    # writer.add_scalar("Macro Avg F1/Eval {}".format(task_name),
                    #                   epoch_metrics[t]['macro avg']['f1-score'],
                    #                   epoch)
                    # writer.add_scalar("Running loss task/Eval {}".format(task_name),
                    #                   task_running_losses[t] / step,
                    #                   epoch)
                    mats.append(mat)

        # for t in range(len(task_predictions)):
        #     mat = []
        #     print(task_list[t].output_labels)
        #     if task_list[t].output_module == "softmax":
        #         mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
        #     elif task_list[t].output_module == "sigmoid":
        #         mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
        #     print(mat)

        training_results.flush_writer()
        training_results.close_writer()
        print('Wrote Evaluation Results')

    @staticmethod
    def calculate_labels(output_module, output):
        if output_module == 'softmax':
            return torch.max(output, 1)[1]
        return output.float()

    @staticmethod
    def get_actual_output(output, task):
        if task.output_module == "softmax":
            translated_out = torch.max(output, 1)[1]
            return translated_out
        elif task.output_module == "sigmoid":
            translated_out = (output >= 0.5).float()
            return torch.squeeze(translated_out)

    @staticmethod
    def get_actual_labels(target, task):
        if task.output_module == "softmax":
            translated_target = torch.max(target, 1)[1]
            return translated_target
        elif task.output_module == "sigmoid":
            return torch.squeeze(target)


def plot_multilabel_confusion(conf_matrix, labels):
    fig, ax = plt.subplots(4 * math.floor(len(labels) / 4), 4 - len(labels) % 4)

    for axes, cfs_matrix, label in zip(ax.flatten(), conf_matrix, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    # fig.tight_layout()
    # plt.show()
    return fig


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
