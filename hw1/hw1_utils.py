import numpy as np
import torch
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#TODO: indicate whether you want to use Numpy or Pytorch here by modifying variable 'use_pytorch'
# If you want to use Pytorch leave it as default True, otherwise set it to False
use_pytorch = True

def bayes_dataset(split, prefix="bayes"):
    '''
    Arguments:
        split (str): "train" or "test"

    Returns:
        X (S x N LongTensor): features of each object, X[i][j] = 0/1
        y (S LongTensor): label of each object, y[i] = 0/1
    
    '''
    return torch.load(f"{prefix}_{split}.pth")

def bayes_eval(prefix="bayes"):
    import hw1
    X, y = bayes_dataset("train", prefix=prefix)
    if not use_pytorch:
        X, y = X.numpy(), y.numpy()
    D = hw1.bayes_MAP(X, y)
    p = hw1.bayes_MLE(y)
    Xtest, ytest = bayes_dataset("test", prefix=prefix)
    if not use_pytorch:
        Xtest, ytest = Xtest.numpy(), ytest.numpy()
    ypred = hw1.bayes_classify(D, p, Xtest)
    return ypred, ytest

def gaussian_dataset(split, prefix="gaussian"):
    '''
    Arguments:
        split (str): "train" or "test"

    Returns:
        X (S x N LongTensor): features of each object, X[i][j] = 0/1
        y (S LongTensor): label of each object, y[i] = 0/1
    
    '''
    return torch.load(f"{prefix}_{split}.pth")

def gaussian_eval(prefix="gaussian"):
    import hw1
    X, y = gaussian_dataset("train", prefix=prefix)
    if not use_pytorch:
        X, y = X.numpy(), y.numpy()
    mu, sigma2 = hw1.gaussian_MAP(X, y)
    p = hw1.gaussian_MLE(y)
    Xtest, ytest = gaussian_dataset("test", prefix=prefix)
    if not use_pytorch:
        Xtest, ytest = Xtest.numpy(), ytest.numpy()
    ypred = hw1.gaussian_classify(mu, sigma2, p, Xtest)
    return ypred, ytest
