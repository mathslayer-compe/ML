import hw1_utils as utils
# choose the library you want to use
import torch
import numpy as np


# Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (S x N LongTensor / Numpy ndarray): features of each object, X[i][j] = 0/1
        y (S LongTensor  / Numpy ndarray): label of each object, y[i] = 0/1

    Returns:
        D (2 x N Float Tensor / Numpy ndarray): MAP estimation of P(X_j=1|Y=i)

    '''
    S, N = X.shape
    D = torch.zeros(2, N, dtype=torch.float)

    for i in range(2):
        prior = X.float().mean(dim=0)
        X_class = X[y == i]
        for j in range(N):
            num_class1 = (X_class[:, j] == 1).sum().float()
            total = X_class.shape[0]
            likelihood = num_class1 / total
            D[i, j] = (likelihood * prior[j]) / prior[j]

    return D


def bayes_MLE(y):
    '''
    Arguments:
        y (S LongTensor / Numpy ndarray): label of each object

    Returns:
        p (float or scalar Float Tensor / Numpy ndarray): MLE of P(Y=1)

    '''
    return y.float().mean()


def bayes_classify(D, p, X):
    '''
    Arguments:
        D (2 x N Float Tensor / Numpy ndarray): returned value of `bayes_MAP`
        p (float or scalar Float Tensor / Numpy ndarray): returned value of `bayes_MLE`
        X (S x N LongTensor / Numpy ndarray): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor / Numpy ndarray): label of each object for classification, y[i] = 0/1

    '''
    S, N = X.shape
    y = torch.zeros(S, dtype=torch.long)

    for i in range(S):
        x = X[i]
        p0 = torch.log(1 - p)
        p1 = torch.log(p)

        for j in range(N):
            if x[j] == 0:
                p0 = p0 + torch.log(1 - D[0, j])
                p1 = p1 + torch.log(1 - D[1, j])

            else:
                p0 = p0 + torch.log(D[0, j])
                p1 = p1 + torch.log(D[1, j])

        if p1 > p0:
            y[i] = 1

        else:
            y[i] = 0

    return y


def gaussian_MAP(X, y):
    '''
    Arguments:
        X (S x N FloatTensor / Numpy ndarray): features of each object
        y (S LongTensor / Numpy ndarray): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor / Numpy ndarray): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor / Numpy ndarray): MAP estimation of mu in N(mu, sigma2)

    '''
    S, N = X.shape
    mu = torch.zeros(2, N, dtype=torch.float)
    sigma2 = torch.zeros(2, N, dtype=torch.float)

    for i in range(2):
        mu[i] = X[y == i].mean(dim=0)
        sigma2[i] = ((X[y == i] - mu[i]) ** 2).mean(dim=0)

    return mu, sigma2


def gaussian_MLE(y):
    '''
    Arguments:
        y (S LongTensor / Numpy ndarray): label of each object

    Returns:
        p (float or scalar Float Tensor / Numpy ndarray): MLE of P(Y=1)

    '''
    return y.float().mean()


def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor / Numpy ndarray): returned value #1 of `gaussian_MAP` (estimation of mean)
        sigma2 (2 x N Float Tensor / Numpy ndarray): returned value #2 of `gaussian_MAP` (square of sigma)
        p (float or scalar Float Tensor / Numpy ndarray): returned value of `bayes_MLE`
        X (S x N LongTensor / Numpy ndarray): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor / Numpy ndarray): label of each object for classification, y[i] = 0/1

    '''
    S, N = X.shape
    y = torch.zeros(S, dtype=torch.long)

    for i in range(S):
        x = X[i]
        vals = []

        for j in range(2):
            prob_sum = -0.5 * torch.log(2 * np.pi * sigma2[j])
            prob_mean = -0.5 * ((x - mu[j]) ** 2) / sigma2[j]
            prob = prob_sum + prob_mean

            if j == 0:
                prior = 1 - p

            else:
                prior = p

            posterior = prob.sum() + torch.log(prior)
            vals.append(posterior)

        y[i] = torch.argmax(torch.tensor(vals))

    return y
