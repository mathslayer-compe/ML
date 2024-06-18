import torch
import hw2_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

'''


# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    n, d = X.size()
    ones_col = torch.ones(n, 1)
    X_b = torch.cat((ones_col, X), dim=1)
    w = torch.zeros(d + 1, 1, dtype=X.dtype)

    for _ in range(num_iter):
        grad = torch.zeros(d + 1, 1, dtype=X.dtype)

        for i in range(n):
            xi = X_b[i].reshape(1, -1)
            yi = Y[i]
            pred = torch.matmul(xi, w)
            error = pred - yi
            grad += 2 * xi.t() * error

        w -= lrate * grad / n

    return w


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    n, d = X.size()
    ones_col = torch.ones(n, 1)
    X_b = torch.cat((ones_col, X), dim=1)
    w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X_b.t(), X_b)), X_b.t()), Y)
    return w


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    w = linear_normal(X, Y)
    plt.scatter(X, Y)
    plt.plot(X, X * w[1] + w[0])

    plt.show()


#plot_linear()


# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    n, d = X.size()
    w = torch.zeros(d + 1, 1)
    prepend_ones = torch.ones(n, 1)
    prepend_X = torch.cat((prepend_ones, X), 1)

    for i in range(num_iter):
        exponent = -Y * torch.matmul(prepend_X, w)
        grad = (torch.exp(exponent) / (2 + torch.exp(exponent))) * (-Y * prepend_X) * (1 / n)
        grad = torch.sum(grad, 0)
        grad = torch.reshape(grad, (d + 1, 1))
        w = w - grad * lrate

    return w


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_logistic_data()
    w_log = logistic(X, Y, 0.01, 1000)
    w_lin = linear_gd(X, Y, 0.01, 1000)
    X_log = X[:, 0:1]
    X_lin = X[:, 1:]
    plt.scatter(X_log, X_lin)
    plt.plot(X_log, -(w_log[0] + w_log[1] * X_log) / w_log[2])
    plt.plot(X_lin, -(w_lin[0] + w_lin[1] * X_lin) / w_lin[2], color='red')
    plt.legend(["Scattered Data", "Logistic Regression", "Linear Regression"])
    plt.title("Logistic Regression vs. Linear Regression")
    plt.xlabel("Logistic")
    plt.ylabel("Linear")

    plt.show()

logistic_vs_ols()
