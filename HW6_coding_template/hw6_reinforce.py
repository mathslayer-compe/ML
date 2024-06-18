import torch
import tqdm
import numpy as np


def mle(y, n_iters):
    """Maximum likelihood estimation given dataset y encoded in delta"""
    alpha = 1
    theta = torch.randn(6)

    delta = torch.zeros(y.numel(), 6).scatter(1, y, torch.ones_like(y).float())

    for iter in tqdm.tqdm(range(n_iters)):
        p_theta = torch.nn.Softmax(dim=0)(theta)
        g = torch.mean(p_theta - delta, 0)
        theta = theta - alpha * g

    return theta


def reinforce(R, n_iters, theta=None):
    """REINFORCE with given reward function."""
    alpha = 1

    if theta is None:
        theta = torch.randn(6)

    for i in tqdm.tqdm(range(n_iters)):

        # current distribution
        p_theta = torch.nn.Softmax(dim=0)(theta)

        # sample from current distribution and compute reward

        # TODO: sample from p_theta, [#samples, 1]
        sample_y = torch.multinomial(p_theta, 1000, replacement=True)
        sample_y = torch.reshape(sample_y, (1000, 1))

        # TODO: use your equation from 4(d) to compute gradient
        delta = torch.zeros(sample_y.numel(), 6).scatter(1, sample_y, torch.ones_like(sample_y).float())
        g = torch.mean(R[sample_y] * (delta - p_theta), 0)

        # update the parameter
        theta = theta + alpha * g

    return theta


if __name__ == "__main__":

    np.random.seed(1)

    p_gt = torch.Tensor([1.0 / 12, 2.0 / 12, 3.0 / 12, 3.0 / 12, 2.0 / 12, 1.0 / 12])
    y = (
        torch.from_numpy(np.random.choice(list(range(6)), size=1000, p=p_gt.numpy()))
        .type(torch.int64)
        .view(-1, 1)
    )

    n_iters = 10000

    theta_mle = mle(y, n_iters)
    R = p_gt
    theta_rl = reinforce(R, n_iters)

    print(p_gt)
    print(torch.nn.Softmax(dim=0)(theta_mle))
    print(torch.nn.Softmax(dim=0)(theta_rl))