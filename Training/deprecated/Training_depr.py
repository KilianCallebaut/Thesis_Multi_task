import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import ConcatDataset

from Tasks.ConcatTaskDataset import ConcatTaskDataset


class Training:

    @staticmethod
    def get_accuracy(model, data):
        loader = torch.utils.data.DataLoader(data, batch_size=500)
        correct, total = 0, 0
        for xs, ts in loader:
            xs = xs.view(-1, 784)  # flatten the image
            zs = model(xs)
            pred = zs.max(1, keepdim=True)[1]  # get the index of the max logit
            correct += pred.eq(ts.view_as(pred)).sum().item()
            total += int(ts.shape[0])
        return correct / total

    # Source: https://www.cs.toronto.edu/~lczhang/321/tut/tut04.pdf
    # Also helpful: https://github.com/sugi-chan/pytorch_multitask/blob/master/pytorch%20multi-task-Copy2.ipynb
    @staticmethod
    def run_gradient_descent(model: nn.Module,
                             concat_dataset: ConcatTaskDataset,
                             batch_size=64,
                             learning_rate=0.01,
                             weight_decay=0,
                             num_epochs=50):
        criteria = [nn.CrossEntropyLoss() for _ in concat_dataset.datasets]
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


        iters, losses = [], []
        iters_sub, train_acc, val_acc = [], [], []

        # Load Data Example
        train_loader = torch.utils.data.DataLoader(
            concat_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

        task_list = [x.name for x in concat_dataset.datasets]

        # training

        # xs pixels, ts targets
        n = 0  # the number of iterations
        for epoch in range(num_epochs):
            print('Epoch {}'.format(epoch))
            print('===========================================================')

            model.train()  # Set model to training mode

            # losses to 0
            running_losses = [0.0 for _ in concat_dataset.datasets]
            running_losses_overall = 0.0

            # corrects to 0
            epoch_corrects = [0.0 for _ in concat_dataset.datasets]
            epoch_wrongs = [0.0 for _ in concat_dataset.datasets]
            epoch_accuracies = [0.0 for _ in concat_dataset.datasets]

            # iterate over data
            for inputs, labels, names in train_loader:

                batch_task_indexes = [task_list.index(name) for name in names]

                # tensors for filtering instances in batch that are not from the task
                # label_flags = [Tensor([1.0 if t == n else 0 for n in names]) for t in task_list]
                # output_flags = [Tensor([[1 for _ in labels[0]] if t == n else [0 for _ in labels[0]] for n in names]) for t in task_list]
                batch_flags = [Tensor([True if t == n else False for n in names]).type(torch.bool) for t in task_list]
                target_flags = [Tensor([0 for _ in x.pad_before] + [1 for _ in x.targets[0]] + [0 for _ in x.pad_after]).type(torch.bool) for x in concat_dataset.datasets]

                losses_batch = [0.0 for _ in task_list]

                # define .cuda() on dataloader(s) to make it run on gpu
                inputs = inputs.cuda()
                labels = labels.cuda()
                # optimizer.zero_grad() to zero parameter gradients
                optimizer.zero_grad()

                output = model(inputs)

                start_idx = 0
                for i in range(len(task_list)):
                    # new_idx = start_idx + len(concat_dataset.datasets[i].targets[0])
                    # bb = output[ label_flags[i].type(torch.bool), :]
                    # criteria[i](output[range(start_idx, new_idx)], )
                    # start_idx = new_idx
                    # filtered_output_1 = output_flags[i][:, range(start_idx, new_idx)] * output[:, range(start_idx, new_idx)]
                    # filtered_labels_1 = torch.mul(label_flags[i], torch.max(labels.float(), 1)[1])


                    filtered_output = output[:, target_flags[i]]
                    filtered_output = filtered_output[batch_flags[i], :]

                    if len(filtered_output) == 0:
                        continue

                    filtered_labels = labels[:, target_flags[i]]
                    filtered_labels = filtered_labels[batch_flags[i], :]
                    filtered_labels = torch.max(filtered_labels.float(), 1)[1].type(torch.LongTensor)
                    filtered_labels = filtered_labels.cuda()
                    losses_batch[i] = criteria[i](filtered_output, filtered_labels)

                    # Statistics
                    epoch_corrects[i] += torch.sum(torch.max(filtered_output, 1)[1] == filtered_labels)
                    epoch_wrongs[i] += torch.sum(torch.max(filtered_output, 1)[1] != filtered_labels)
                    running_losses[i] += losses_batch[i].item() * len(torch.max(filtered_output, 1)[1])

                loss = sum(losses_batch)
                loss.backward()
                optimizer.step()

                # Statistics
                # for i in range(len(task_list)):
                #     filtered_output = output[:, target_flags[i]]
                #     filtered_output = filtered_output[batch_flags[i], :]
                #     if len(filtered_output) == 0:
                #         continue
                #     filtered_output = torch.max(filtered_output, 1)[1]
                #
                #     filtered_labels = labels[:, target_flags[i]]
                #     filtered_labels = filtered_labels[batch_flags[i], :]
                #     filtered_labels = torch.max(filtered_labels.float(), 1)[1]
                #
                #     epoch_corrects[i] += torch.sum(filtered_output == filtered_labels)
                #     epoch_wrongs[i] += torch.sum(filtered_output != filtered_labels)
                #     running_losses[i] += losses_batch[i].item() * len(filtered_output)

                # Statistics
                running_losses_overall += loss.item() * inputs.size(0)

                # print("hey")

            epoch_acc_overall = sum(epoch_corrects).double() / sum([x.__len__() for x in concat_dataset.datasets])
            acc_string = 'Accuracies: Overall: {}, '.format(epoch_acc_overall)

            epoch_losses = [0.0 for _ in concat_dataset.datasets]
            epoch_losses_overall = running_losses_overall / sum([x.__len__() for x in concat_dataset.datasets])
            loss_string = 'Losses: Overall: {}, '.format(epoch_losses_overall)

            for i in range(len(task_list)):
                epoch_accuracies[i] = epoch_corrects[i].double() / concat_dataset.datasets[i].__len__()
                acc_string += task_list[i] + ' accuracy: {}'.format(epoch_accuracies[i]) + ', '
                epoch_losses[i] = running_losses[i] / concat_dataset.datasets[i].__len__()
                loss_string += task_list[i] + ' loss: {}, '.format(epoch_losses[i])

            print()
            print(acc_string)
            print(loss_string)



            # outputs = model(data)
            # for each separate task
            # loss = criterion(output, target)
            # loss_total = loss1 +loss2 +...
            # loss_total.backward()
            # optimizer.step()
            # save the current training information
            #     iters.append(n)
            #     losses.append(float(loss) / batch_size)  # compute *average* loss

            # for xs, ts in iter(dataset):
            #     if len(ts) != batch_size:
            #         continue
            #     # xs = xs.view(-1, 784)  # flatten the image. The -1 is a wildcard
            #     zs = model(xs)
            #     loss = criterion(zs, ts)  # compute the total loss
            #     loss.backward()  # compute updates for each parameter
            #     optimizer.step()  # make the updates for each parameter
            #     optimizer.zero_grad()  # a clean up step for PyTorch
            #     # save the current training information
            #     iters.append(n)
            #     losses.append(float(loss) / batch_size)  # compute *average* loss
            #     if n % 10 == 0:
            #         iters_sub.append(n)
            #         train_acc.append(Training.get_accuracy(model, dataset))
            #         val_acc.append(Training.get_accuracy(model, dataset_val))
            #     # increment the iteration number
            #     n += 1
        return model
