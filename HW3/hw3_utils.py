import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.datasets import load_digits
from os import listdir
import re
from PIL import Image
from torchvision import transforms

''' Start SVM helpers '''

'''
    Produces a contour plot for the prediction function.

    Arguments:
        pred_fxn: Prediction function that takes an n x d tensor of test examples
        and returns your SVM's predictions.
        xmin: Minimum x-value to plot.
        xmax: Maximum x-value to plot.
        ymin: Minimum y-value to plot.
        ymax: Maximum y-value to plot.
        ngrid: Number of points to be plotted between max and min (granularity).
'''
def svm_contour(pred_fxn, xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33):
    '''Plot the contour lines of the svm predictor. '''
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)),
            dim = 2).view(-1, 2)
        zz = pred_fxn(x_test)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                         cmap = 'coolwarm')
        plt.clabel(cs)
        plt.show()

def xor_data():
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y

''' End SVM Helpers '''

''' Start CNN Helpers '''

def torch_digits():
    '''
    Get the training and test datasets for your convolutional neural network
    @return train, test: two torch.utils.data.Datasets
    '''
    digits, labels = load_digits(return_X_y=True)
    digits = torch.tensor(np.reshape(digits, [-1, 8, 8]), dtype=torch.float)
    labels = torch.tensor(np.reshape(labels, [-1]), dtype=torch.long)
    test_X = digits[:180,:,:]
    test_Y = labels[:180]
    digits = digits[180:,:,:]
    labels = labels[180:]
    train = torch.utils.data.TensorDataset(digits, labels)
    test = torch.utils.data.TensorDataset(test_X, test_Y)
    return train, test

def epoch_loss(net, loss_func, data_loader):
    ''' Computes the loss of the model on the entire dataset given by the dataloader.
    Be sure to wrap this function call in a torch.no_grad statement to prevent
    gradient computation.

    @param net: The neural network to be evaluated
    @param loss_func: The loss function used to evaluate the neural network
    @param data_loader: The DataLoader which loads minibatches of the dataset

    @return The network's average loss over the dataset.
    '''
    total_examples = 0
    losses = []
    for X, Y in data_loader:
        total_examples += len(X)
        losses.append(loss_func(net(X), Y).item() * len(X)) # Compute total loss for batch

    return torch.tensor(losses).sum() / total_examples

def train_batch(net, loss_func, xb, yb, opt=None):
    ''' Performs a step of optimization.

    @param net: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer used to improve the model.
    '''
    loss = loss_func(net(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
''' End CNN Helpers '''