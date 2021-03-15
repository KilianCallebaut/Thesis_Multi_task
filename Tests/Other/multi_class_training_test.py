import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms


# load the training data
mnist_train = datasets.MNIST('data', train=True, download=True)
mnist_train = list(mnist_train)[:2500]

import matplotlib.pyplot as plt
for k, (image, label) in enumerate(mnist_train[:18]):
    plt.subplot(3, 6, k+1)
    plt.imshow(image, cmap='gray')
    # transform the image data type to a 28x28 matrix of numbers
    img_to_tensor = transforms.ToTensor()
    # convert the last image we saw into a tensor
    img_tensor = img_to_tensor(image)
    img_tensor.shape

mnist_train = datasets.MNIST('data', train=True, transform=img_to_tensor)
mnist_train = list(mnist_train)[:2500]
mnist_train, mnist_val = mnist_train[:2000], mnist_train[2000:]


def run_gradient_descent(model,
                         batch_size=64,
                         learning_rate=0.01,
                         weight_decay=0,
                         num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iters, losses = [], []
    iters_sub, train_acc, val_acc = [], [], []
    train_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True)
    # training
    n = 0  # the number of iterati
    for epoch in range(num_epochs):
        for xs, ts in iter(train_loader):
            if len(ts) != batch_size:
                continue
        xs = xs.view(-1, 784)  # flatten the image. The -1 is a wildcard
        zs = model(xs)
        loss = criterion(zs, ts)  # compute the total loss
        loss.backward()  # compute updates for each parameter
        optimizer.step()  # make the updates for each parameter
        optimizer.zero_grad()  # a clean up step for PyTorch
        # save the current training information
        iters.append(n)
        losses.append(float(loss) / batch_size)  # compute *average* loss
        if n % 10 == 0:
            iters_sub.append(n)
            train_acc.append(get_accuracy(model, mnist_train))
            val_acc.append(get_accuracy(model, mnist_val))
        # increment the iteration number
        n += 1
    # plotting
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters_sub, train_acc, label="Train")
    plt.plot(iters_sub, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    return model


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


model = nn.Linear(784, 10)
run_gradient_descent(model, batch_size=64, learning_rate=0.01, num_epochs=10)
