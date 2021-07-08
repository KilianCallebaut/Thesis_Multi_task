import datetime

import torch
import torch.optim as optim
from sklearn import metrics
from torch import nn
from torch.utils.data import ConcatDataset

from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.Samplers.MultiTaskSampler import MultiTaskSampler
from Training.Results import Results
from Training.TrainingUtils import TrainingUtils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Training:

    @staticmethod
    def create_results(modelname, task_list, num_epochs, model_checkpoints_path=None, fold=None):
        # run_name creation
        run_name = "Result_" + str(
            datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")) + "_" + modelname
        for n in task_list:
            run_name += "_" + n.name
        if fold:
            run_name += "_fold_{}".format(fold)
        results = Results(model_checkpoints_path=model_checkpoints_path,
                          run_name=run_name, num_epochs=num_epochs)
        return results

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
                             test_dataset=None,
                             optimizer=None,
                             train_loader=None,
                             training_utils=None):
        task_list = concat_dataset.get_task_list()
        n_tasks = len(task_list)

        criteria = [t.loss_function for t in task_list]

        if not results:
            results = Training.create_results(
                modelname=model.name,
                task_list=task_list,
                num_epochs=num_epochs
            )

        if not train_loader:
            train_loader = torch.utils.data.DataLoader(
                concat_dataset,
                num_workers=0,
                pin_memory=False,
                batch_sampler=MultiTaskSampler(dataset=concat_dataset, batch_size=batch_size)
            )
            print(len(train_loader))

        if not optimizer:
            # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if not training_utils:
            training_utils = TrainingUtils()

        #
        target_flags = concat_dataset.get_target_flags()

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
            for inputs, labels, groups in train_loader:
                # tensors for filtering instances in batch and targets that are not from the task
                batch_flags = [[True if t.task_group == g else False for g in groups] for t in
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
                    perc_s = 'I' * (perc - len(str(perc))) + str(perc) + '%'
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
                    filtered_labels = task_list[i].translate_labels(filtered_labels)

                    losses_batch[i] = criteria[i](filtered_output, filtered_labels)
                    output_batch[i] = filtered_output.detach()
                    labels_batch[i] = filtered_labels.detach()
                    del filtered_output
                    del filtered_labels

                # training step
                loss = training_utils.combine_loss(losses_batch)

                loss.backward()
                optimizer.step()

                # Statistics

                step += 1

                for t in range(len(task_list)):
                    task_labels[t] += labels_batch[t].tolist()
                    task_predictions[t] += task_list[t].decision_making(output_batch[t]).tolist()

                running_loss += loss.item() * inputs.size(0)
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
                if task_list[t].classification_type == "multi-class":
                    mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                    results.add_confusion_matrix(epoch, mat, task_list[t], True)
                elif task_list[t].classification_type == "multi-label":
                    mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                    results.add_multi_confusion_matrix(epoch, mat, task_list[t], True)

                print(task_list[t].output_labels)
                mats.append(mat)

            results.add_model_parameters(epoch, model)

            if test_dataset:
                Training.evaluate(model,
                                  test_dataset,
                                  results,
                                  batch_size,
                                  num_epochs=epoch + 1,
                                  start_epoch=epoch,
                                  blank=False)

            if training_utils.early_stop(results=results, epoch=epoch):
                break
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
                 blank=True,
                 eval_loader=None,
                 training_utils=None):

        task_list = concat_dataset.get_task_list()
        n_tasks = len(task_list)
        target_flags = concat_dataset.get_target_flags()

        criteria = [t.loss_function for t in task_list]

        if not eval_loader:
            eval_loader = torch.utils.data.DataLoader(
                concat_dataset,
                num_workers=0,
                pin_memory=False,
                batch_sampler=MultiTaskSampler(dataset=concat_dataset, batch_size=batch_size)
            )

        if not training_utils:
            training_utils = TrainingUtils()

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

                for inputs, labels, groups in eval_loader:
                    batch_flags = [[True if t.task_group == g else False for g in groups] for t in
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
                        perc_s = 'I' * (perc - len(str(perc))) + str(perc) + '%'
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

                        filtered_labels = task_list[i].translate_labels(filtered_labels)
                        losses_batch[i] = criteria[i](filtered_output, filtered_labels)
                        output_batch[i] = filtered_output.detach()
                        labels_batch[i] = filtered_labels.detach()

                    loss = training_utils.combine_loss(losses_batch)

                    running_loss += loss.item() * inputs.size(0)
                    step += 1
                    for t in range(len(task_list)):
                        task_labels[t] += labels_batch[t].tolist()
                        task_predictions[t] += task_list[t].decision_making(output_batch[t]).tolist()

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

                    if task_list[t].classification_type == "multi-class":
                        mat = metrics.confusion_matrix(task_labels[t], task_predictions[t])
                        training_results.add_confusion_matrix(epoch, mat, task_list[t], False)
                    elif task_list[t].classification_type == "multi-label":
                        mat = metrics.multilabel_confusion_matrix(task_labels[t], task_predictions[t])
                        training_results.add_multi_confusion_matrix(epoch, mat, task_list[t], False)
                    print(task_list[t].output_labels)
                    mats.append(mat)

        training_results.flush_writer()
        print('Wrote Evaluation Results')
