import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def linear_kernel(x, y):
    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )

    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        return torch.dot(x, y)


def polynomial_kernel(x, y, p):
    '''
    Compute the polynomial kernel function with arbitrary power p

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        p: the power of the polynomial kernel

    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        return (torch.dot(x, y) + 1) ** p


def gaussian_kernel(x, y, sigma):
    '''
    Compute the linear kernel function

    Arguments:
        x: 1d tensor with shape (d, )
        y: 1d tensor with shape (d, )
        sigma: parameter sigma in rbf kernel

    Returns:
        a torch.float32 scalar
    '''
    with torch.no_grad():
        return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))


def svm_epoch_loss(alpha, x_train, y_train, kernel=linear_kernel):
    '''
    Compute the linear kernel function

    Arguments:
        alpha: 1d tensor with shape (N,), alpha is the trainable parameter in our svm
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Returns:
        a torch.float32 scalar which is the loss function of current epoch
    '''
    with torch.no_grad():
        N = x_train.shape[0]
        loss = 0
        for i in range(N):
            for j in range(N):
                loss += alpha[i] * alpha[j] * y_train[i] * y_train[j] * kernel(x_train[i], x_train[j])
        loss = 0.5 * loss - alpha.sum()
        return loss


def svm_solver(x_train, y_train, lr, num_iters,
               kernel=linear_kernel, c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is the linear kernel.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    num_samples = x_train.size(0)
    alpha = torch.zeros(num_samples, requires_grad=True)

    for i in range(num_iters):
        dest = 0

        for j in range(num_samples):
            for k in range(num_samples):
                dest += 0.5 * alpha[j] * alpha[k] * y_train[j] * y_train[k] * kernel(x_train[j, :], x_train[k, :])

            dest -= alpha[j]

        dest.backward()

        with torch.no_grad():
            alpha -= lr * alpha.grad
            if c is not None:
                alpha.clamp_(0, c)

            else:
                alpha.clamp_(min=0)

        alpha.grad.zero_()

    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=linear_kernel):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    idx = torch.nonzero(alpha, as_tuple=True)
    alpha_ = alpha[idx]
    x_ = x_train[idx]
    y_ = y_train[idx]
    if len(alpha_) == 0:
        return torch.zeros((x_test.shape[0],))
    id = alpha_.argmin()
    b = 1 / y_[id]
    for i in range(len(alpha_)):
        b -= alpha_[i] * y_[i] * kernel(x_[i], x_[id])
    y_test = torch.zeros((x_test.shape[0],))
    for j in range(len(x_test)):
        y_test[j] = b
        for i in range(len(alpha_)):
            y_test[j] += alpha_[i] * y_[i] * kernel(x_[i], x_test[j])
    return y_test


class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0)  # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3)
        self.conv2 = nn.Conv2d(7, 3, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=2)
        self.fc = nn.Linear(3 * 2 * 2, 10)

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(7)
        self.batch_norm2 = nn.BatchNorm2d(3)
        self.batch_norm3 = nn.BatchNorm2d(3)

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb = xb.unsqueeze(1)

        x = F.relu(self.batch_norm1(self.conv1(xb)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class DigitsConvNetv2(nn.Module):
    def __init__(self):
        super(DigitsConvNetv2, self).__init__()
        torch.manual_seed(0)  # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=2)
        self.fc1 = nn.Linear(10 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

        self.batch_norm1 = nn.BatchNorm2d(10)
        self.batch_norm2 = nn.BatchNorm2d(10)
        self.batch_norm3 = nn.BatchNorm2d(10)
        self.dropout = nn.Dropout(0.07)

    def forward(self, xb):
        xb = xb.unsqueeze(1)

        x = F.relu(self.batch_norm1(self.conv1(xb)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    net.eval()
    with torch.no_grad():
        training_loss = hw3_utils.epoch_loss(net, loss_func, train_dl)
        testing_loss = hw3_utils.epoch_loss(net, loss_func, test_dl)
        train_losses.append(training_loss)
        test_losses.append(testing_loss)

    for epoch in range(n_epochs):
        net.train()
        for i, (images, labels) in enumerate(train_dl):
            hw3_utils.train_batch(net, loss_func, images, labels, optimizer)

        net.eval()
        with torch.no_grad():
            training_loss = hw3_utils.epoch_loss(net, loss_func, train_dl)
            testing_loss = hw3_utils.epoch_loss(net, loss_func, test_dl)
            train_losses.append(training_loss)
            test_losses.append(testing_loss)

    return train_losses, test_losses


model = DigitsConvNet()
train, test = hw3_utils.torch_digits()
fcn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.005)
train_losses, test_losses = fit_and_evaluate(model, optim, fcn, train, test, 30, 8)
epoch = np.linspace(0, 30, 31)
plt.title('Training Loss vs. Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Amount of Loss')
plt.plot(epoch, train_losses, label='Training Loss')
plt.plot(epoch, test_losses, color='blue', label='Testing Loss')
plt.legend()
plt.show()
