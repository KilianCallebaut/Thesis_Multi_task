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
                             num_epochs=50):
        criteria = [nn.BCELoss() if d.task.output_module == 'sigmoid' else nn.CrossEntropyLoss()
                    for d in concat_dataset.datasets]

        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        results = Results(concat_dataset=concat_dataset, batch_size=batch_size, learning_rate=learning_rate,
                          weight_decay=weight_decay, nr_epochs=num_epochs)
        task_list = [x.task for x in concat_dataset.datasets]

        name = ''
        for n in task_list:
            name += "_" + n.name
        writer = SummaryWriter(comment=name)

        writer.add_graph(model, torch.rand([1, 168]).cuda())
        for tc in concat_dataset.datasets:
            writer.add_embedding(torch.stack(tc.inputs), tc.targets)

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
                elif task_list[t].output_module == "sigmoid":
                    writer.add_scalar("Micro AVG F1/{}".format(task_name),
                                      epoch_metrics[t]['micro avg']['f1-score'], epoch)

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
            if task_list[t].output_module == "softmax":
                print(task_list[t].output_labels)
                print(metrics.confusion_matrix(task_labels[t], task_predictions[t]))
            elif task_list[t].output_module == "sigmoid":
                print(task_list[t].output_labels)
                print(metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t]))

        writer.flush()
        writer.close()
        print('Wrote Training Results')

        return model, results

    # @staticmethod
    # def evaluate_in_run(model: nn.Module,
    #                     eval_dataset: ConcatTaskDataset,
    #                     batch_size=64,
    #                     epoch=0,
    #                     writer=SummaryWriter):
    #     model.eval()

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

        writer = SummaryWriter(comment=name+'_evaluation')
        for tc in concat_dataset:
            writer.add_embedding(torch.tensor(tc.inputs), tc.targets)

        eval_loader = torch.utils.data.DataLoader(
            concat_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        print("Start Evaluation")
        for epoch in range(start_epoch, num_epochs):
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

                if task_list[t].output_module == "softmax":
                    writer.add_scalar("Accuracy/Eval {}".format(task_name), epoch_metrics[t]['accuracy'], epoch)
                elif task_list[t].output_module == "sigmoid":
                    writer.add_scalar("Micro AVG F1/Eval {}".format(task_name),
                                      epoch_metrics[t]['micro avg']['f1-score'], epoch)

                writer.add_scalar("Macro Avg Precision/Eval {}".format(task_name),
                                  epoch_metrics[t]['macro avg']['precision'], epoch)
                writer.add_scalar("Macro Avg F1/Eval {}".format(task_name), epoch_metrics[t]['macro avg']['f1-score'],
                                  epoch)
                writer.add_scalar("Running loss task/Eval {}".format(task_name), task_running_losses[t].item() / step,
                                  epoch)

        for t in range(len(task_predictions)):
            if task_list[t].output_module == "softmax":
                print(task_list[t].output_labels)
                print(metrics.confusion_matrix(task_labels[t], task_predictions[t]))

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

    # tasks, dict

    # def calculate_metrics_epoch_per_task(self, nr_epoch):
    #     pred_target_per_task = self.calculate_true_predicted_epoch_per_task(nr_epoch)
    #     return [metrics.classification_report([pred_tar[1].tolist() for pred_tar in pred_target_per_task[task_idx]],
    #                                           [pred_tar[0].tolist() for pred_tar in pred_target_per_task[task_idx]],
    #                                           target_names=self.concat_dataset.get_task_list()[
    #                                               task_idx].output_labels,
    #                                           output_dict=True)
    #             for task_idx in range(len(pred_target_per_task))]
