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

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.Samplers.MultiTaskSampler import MultiTaskSampler
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

        criteria = [nn.BCELoss().to(device) if t.output_module == 'sigmoid' else nn.CrossEntropyLoss().to(device)
                    for t in task_list]
        target_flags = [
            [False for _ in x.pad_before] + [True for _ in x.targets[0]] + [False for _ in x.pad_after]
            for x in datasets]
        train_loader = torch.utils.data.DataLoader(
            concat_dataset,
            num_workers=0,
            pin_memory=True,
            batch_sampler=MultiTaskSampler(dataset=concat_dataset, batch_size=batch_size)
        )
        print(len(train_loader))

        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Epoch
        for epoch in range(start_epoch, num_epochs):
            model.train()  # Set model to training mode

            print('Epoch {}'.format(epoch))
            print('===========================================================')

            running_loss = 0.0
            step = 0
            task_predictions = [[] for _ in task_list]
            task_labels = [[] for _ in task_list]
            task_running_losses = [0 for _ in task_list]

            perc = 0
            begin = datetime.datetime.now()
            ex_mx = datetime.timedelta(0)
            ex_t = datetime.timedelta(0)
            # iterate over data
            for inputs, labels, names in train_loader:

                # if len(labels) != batch_size:
                #     continue

                # tensors for filtering instances in batch and targets that are not from the task
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
                    while perc < (step / len(train_loader)) * 100:
                        perc += 1
                    perc_s = 'I' * perc
                    perc_sp = ' ' * (100 - perc)
                    ex = datetime.datetime.now() - begin
                    begin = datetime.datetime.now()
                    ex_mx = ex if ex > ex_mx else ex_mx
                    ex_t += ex
                    print('[{}{}], execution time: {}, max time: {}, total time: {}'.format(perc_s, perc_sp, ex, ex_mx,
                                                                                            ex_t),
                          end='\r' if perc != 100 else '\n')

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
                    mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                    results.add_confusion_matrix(epoch, mat, task_list[t], True)
                elif task_list[t].output_module == "sigmoid":
                    mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                    results.add_multi_confusion_matrix(epoch, mat, task_list[t], True)

                print(task_list[t].output_labels)
                mats.append(mat)

            results.add_model_parameters(epoch, model)

            if 'test_dataset' in kwargs:
                Training.evaluate(model,
                                  kwargs.get('test_dataset'),
                                  results,
                                  batch_size,
                                  num_epochs=epoch + 1,
                                  start_epoch=epoch,
                                  blank=False)
        print('Training Done')
        results.flush_writer()
        print('Wrote Training Results')

        return model, results

    @staticmethod
    def evaluate(blank_model: nn.Module,
                 concat_dataset: ConcatTaskDataset,
                 training_results: Results,
                 batch_size=64,
                 num_epochs=50,
                 start_epoch=0,
                 blank=True):

        datasets = concat_dataset.datasets
        task_list = [x.task for x in datasets]
        n_tasks = len(task_list)

        criteria = [nn.BCELoss().to(device) if d.task.output_module == 'sigmoid' else nn.CrossEntropyLoss().to(device)
                    for d in datasets]
        target_flags = [
            [False for _ in x.pad_before] + [True for _ in x.targets[0]] + [False for _ in x.pad_after]
            for x in datasets]
        eval_loader = torch.utils.data.DataLoader(
            concat_dataset,
            num_workers=0,
            pin_memory=True,
            batch_sampler=MultiTaskSampler(dataset=concat_dataset, batch_size=batch_size)
        )

        blank_model.eval()

        with torch.no_grad():
            print("Start Evaluation")
            for epoch in range(start_epoch, num_epochs):
                print('Epoch {}'.format(epoch))
                print('===========================================================')

                if blank:
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

                    batch_flags = [[True if t.name == n else False for n in names] for t in
                                   task_list]

                    losses_batch = [torch.tensor([0]).to(device) for _ in task_list]
                    output_batch = [torch.Tensor([]) for _ in task_list]
                    labels_batch = [torch.Tensor([]) for _ in task_list]

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output = blank_model(inputs)

                    if perc < (step / len(eval_loader)) * 100:
                        while perc < (step / len(eval_loader)) * 100:
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
                                                                                              ex_t),
                            end='\r' if perc != 100 else '\n')

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
                        mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                        training_results.add_confusion_matrix(epoch, mat, task_list[t], False)
                    elif task_list[t].output_module == "sigmoid":
                        mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                        training_results.add_multi_confusion_matrix(epoch, mat, task_list[t], False)
                    print(task_list[t].output_labels)
                    mats.append(mat)

        training_results.flush_writer()
        # training_results.close_writer()
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
