import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from sklearn import metrics
from torch import nn, Tensor
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Training.Results import Results


class Training:

    # Source: https://www.cs.toronto.edu/~lczhang/321/tut/tut04.pdf
    # Also helpful: https://github.com/sugi-chan/pytorch_multitask/blob/master/pytorch%20multi-task-Copy2.ipynb
    @staticmethod
    def run_gradient_descent(model: nn.Module,
                             concat_dataset: ConcatTaskDataset,
                             batch_size=64,
                             learning_rate=0.05,
                             weight_decay=0,
                             num_epochs=50,
                             **kwargs):
        criteria = [nn.BCELoss() if d.task.output_module == 'sigmoid' else nn.CrossEntropyLoss()
                    for d in concat_dataset.datasets]

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        results = Results(concat_dataset=concat_dataset, batch_size=batch_size, learning_rate=learning_rate,
                          weight_decay=weight_decay, nr_epochs=num_epochs, **kwargs)
        task_list = [x.task for x in concat_dataset.datasets]

        name = model.name
        for n in task_list:
            name += "_" + n.name
        writer = SummaryWriter(comment=name)

        # writer.add_graph(model, torch.rand([2, 2206, 24]).cuda())

        # Load Data Example
        train_loader = torch.utils.data.DataLoader(
            concat_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        # Epoch
        for epoch in range(num_epochs):

            print('Epoch {}'.format(epoch))
            print('===========================================================')

            running_loss = 0.0
            step = 0
            task_predictions = [[] for _ in task_list]
            task_labels = [[] for _ in task_list]
            task_running_losses = [0 for _ in task_list]

            model.train()  # Set model to training mode

            before = list(model.parameters())[0].clone()
            # iterate over data
            for inputs, labels, names in train_loader:

                # if len(labels) != batch_size:
                #     continue

                # tensors for filtering instances in batch and targets that are not from the task
                batch_flags = [Tensor([True if t.name == n else False for n in names]).type(torch.bool) for t in
                               task_list]
                target_flags = [
                    Tensor([0 for _ in x.pad_before] + [1 for _ in x.targets[0]] + [0 for _ in x.pad_after]).type(
                        torch.bool) for x in concat_dataset.datasets]

                losses_batch = [0.0 for _ in task_list]
                output_batch = [torch.Tensor([]) for _ in task_list]
                labels_batch = [torch.Tensor([]) for _ in task_list]

                # define .cuda() on dataloader(s) to make it run on gpu
                inputs = inputs.cuda()
                labels = labels.cuda()

                # optimizer.zero_grad() to zero parameter gradients
                optimizer.zero_grad()

                # model
                output = model(inputs)

                for i in range(len(task_list)):
                    if sum(batch_flags[i]) == 0:
                        continue

                    filtered_output = output[i]
                    filtered_output = filtered_output[batch_flags[i], :]

                    filtered_labels = labels[:, target_flags[i]]
                    filtered_labels = filtered_labels[batch_flags[i], :]

                    losses_batch[i] = criteria[i](filtered_output,
                                                  Training.calculate_labels(task_list[i].output_module,
                                                                            filtered_labels))
                    output_batch[i] = filtered_output
                    labels_batch[i] = filtered_labels

                # training step
                loss = 0
                for l in range(len(losses_batch)):
                    loss = loss + losses_batch[l]
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                step += 1
                task_labels = [
                    task_labels[t] +
                    [Training.get_actual_labels(l[None, :], task_list[t]).tolist() for l in labels_batch[t]] for
                    t in range(len(labels_batch))]
                task_predictions = [
                    task_predictions[t] +
                    [Training.get_actual_output(l[None, :], task_list[t]).tolist() for l in output_batch[t]] for
                    t in range(len(output_batch))]
                task_running_losses = [task_running_losses[t] + losses_batch[t]
                                       for t in range(len(losses_batch))]
                # results.add_output(epoch, output_batch, labels_batch, losses_batch, loss.item())

            writer.add_scalar("Loss/train", running_loss / step, epoch)
            epoch_metrics = [metrics.classification_report(task_labels[t], task_predictions[t], output_dict=True) for t
                             in range(len(task_predictions))]

            for t in range(len(task_list)):
                task_name = task_list[t].name

                if task_list[t].output_module == "softmax":
                    writer.add_scalar("Accuracy/{}".format(task_name), epoch_metrics[t]['accuracy'], epoch)
                    # mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                    # fig = plt.figure()
                    # plt.imshow(mat)
                    # writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
                    #                   fig, epoch)
                elif task_list[t].output_module == "sigmoid":
                    writer.add_scalar("Micro AVG F1/{}".format(task_name),
                                      epoch_metrics[t]['micro avg']['f1-score'], epoch)
                    # mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                    # fig = plot_multilabel_confusion(mat, task_list[t].output_labels)
                    # writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
                    #                   fig, epoch)

                writer.add_scalar("Macro Avg Precision/{}".format(task_name),
                                  epoch_metrics[t]['macro avg']['precision'], epoch)
                writer.add_scalar("Macro Avg F1/{}".format(task_name), epoch_metrics[t]['macro avg']['f1-score'],
                                  epoch)
                writer.add_scalar("Running loss task/{}".format(task_name), task_running_losses[t].item() / step, epoch)

            for h in range(len(model.hidden)):
                writer.add_histogram("hidden weights {}".format(h), model.hidden[h].weight, epoch)
            for t in range(len(model.task_nets)):
                writer.add_histogram("task_nets weights", model.task_nets[t].weight, epoch)

            after = list(model.parameters())[0].clone()
            print("Not updated parameters?")
            print(torch.equal(before.data, after.data))

            results.add_model_parameters(epoch, model)

        print('Training Done')
        # results.writefiles(True)

        for t in range(len(task_predictions)):
            mat = []
            print(task_list[t].output_labels)
            if task_list[t].output_module == "softmax":
                mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                fig = plt.figure()
                plt.imshow(mat)
                writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
                                  fig, epoch)
            elif task_list[t].output_module == "sigmoid":
                mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                fig = plot_multilabel_confusion(mat, task_list[t].output_labels)
                writer.add_figure('Confusion matrix/{}'.format(task_list[t]),
                                  fig, epoch)
            print(mat)

        writer.flush()
        writer.close()
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
        criteria = [nn.BCELoss() if d.task.output_module == 'sigmoid' else nn.CrossEntropyLoss()
                    for d in concat_dataset.datasets]
        # results = Results(concat_dataset=concat_dataset, batch_size=batch_size, nr_epochs=num_epochs)
        task_list = [x.task for x in concat_dataset.datasets]

        name = ''
        for n in task_list:
            name += "_" + n.name

        writer = SummaryWriter(comment=name + '_evaluation')

        eval_loader = torch.utils.data.DataLoader(
            concat_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        print("Start Evaluation")
        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}'.format(epoch))
            print('===========================================================')

            training_results.load_model_parameters(epoch, blank_model)
            blank_model.eval()

            running_loss = 0.0
            step = 0
            task_predictions = [[] for _ in task_list]
            task_labels = [[] for _ in task_list]
            task_running_losses = [0 for _ in task_list]

            for inputs, labels, names in eval_loader:

                # if len(labels) != batch_size:
                #     continue

                # tensors for filtering instances in batch and targets that are not from the task
                batch_flags = [Tensor([True if t.name == n else False for n in names]).type(torch.bool) for t in
                               task_list]
                target_flags = [
                    Tensor([0 for _ in x.pad_before] + [1 for _ in x.targets[0]] + [0 for _ in x.pad_after]).type(
                        torch.bool) for x in concat_dataset.datasets]

                losses_batch = [0.0 for _ in task_list]
                output_batch = [torch.Tensor([]) for _ in task_list]
                labels_batch = [torch.Tensor([]) for _ in task_list]

                inputs = inputs.cuda()
                labels = labels.cuda()
                output = blank_model(inputs)

                for i in range(len(task_list)):
                    if sum(batch_flags[i]) == 0:
                        continue

                    filtered_output = output[i]
                    filtered_output = filtered_output[batch_flags[i], :]

                    filtered_labels = labels[:, target_flags[i]]
                    filtered_labels = filtered_labels[batch_flags[i], :]

                    losses_batch[i] = criteria[i](filtered_output,
                                                  Training.calculate_labels(task_list[i].output_module,
                                                                            filtered_labels))
                    output_batch[i] = filtered_output
                    labels_batch[i] = filtered_labels

                loss = 0
                for l in range(len(losses_batch)):
                    loss = loss + losses_batch[l]

                running_loss += loss
                step += 1
                task_labels = [
                    task_labels[t] +
                    [Training.get_actual_labels(l[None, :], task_list[t]).tolist() for l in labels_batch[t]] for
                    t in range(len(labels_batch))]
                task_predictions = [
                    task_predictions[t] +
                    [Training.get_actual_output(l[None, :], task_list[t]).tolist() for l in output_batch[t]] for
                    t in range(len(output_batch))]
                task_running_losses = [task_running_losses[t] + losses_batch[t]
                                       for t in range(len(losses_batch))]

            writer.add_scalar("Loss/eval", running_loss / step, epoch)
            epoch_metrics = [metrics.classification_report(task_labels[t], task_predictions[t], output_dict=True) for t
                             in range(len(task_predictions))]

            for t in range(len(task_list)):
                task_name = task_list[t].name
                mat = []

                if task_list[t].output_module == "softmax":
                    writer.add_scalar("Accuracy/Eval {}".format(task_name), epoch_metrics[t]['accuracy'], epoch)
                    mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                    fig = plt.figure()
                    plt.imshow(mat)
                    writer.add_figure('Confusion matrix/Eval {}'.format(task_list[t]),
                                      fig, epoch)
                elif task_list[t].output_module == "sigmoid":
                    writer.add_scalar("Micro AVG F1/Eval {}".format(task_name),
                                      epoch_metrics[t]['micro avg']['f1-score'], epoch)
                    mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                    fig = plot_multilabel_confusion(mat, task_list[t].output_labels)
                    writer.add_figure('Confusion matrix/Eval {}'.format(task_list[t]),
                                      fig, epoch)

                writer.add_scalar("Macro Avg Precision/Eval {}".format(task_name),
                                  epoch_metrics[t]['macro avg']['precision'], epoch)
                writer.add_scalar("Macro Avg F1/Eval {}".format(task_name), epoch_metrics[t]['macro avg']['f1-score'],
                                  epoch)
                writer.add_scalar("Running loss task/Eval {}".format(task_name), task_running_losses[t].item() / step,
                                  epoch)

        for t in range(len(task_predictions)):
            mat = []
            print(task_list[t].output_labels)
            if task_list[t].output_module == "softmax":
                mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
            elif task_list[t].output_module == "sigmoid":
                mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
            print(mat)

        writer.flush()
        writer.close()

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
    fig, ax = plt.subplots(4 * math.floor(len(labels)/4), 4 - len(labels) % 4)

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
