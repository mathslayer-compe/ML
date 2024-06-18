import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    N = 20
    std = .7
    torch.manual_seed(5)
    x = torch.cat(
        (
            std * torch.randn(N, 2) + torch.Tensor([[2, -2]]),
            std * torch.randn(N, 2) + torch.Tensor([[-2, 2]]),
        ),
        0,
    )
    init_c = torch.Tensor([[-2, 2], [-2, 2]]) + std * torch.randn(2, 2)
    return x, init_c


def vis_cluster(c, x1, x2):
    '''
    Visualize the data and clusters.

    Argument:
        c: cluster centers [2, 2]
        x1: data points belonging to cluster 1 [#cluster_points, 2]
        x2: data points belonging to cluster 2 [#cluster_points, 2]
    '''
    # c[2, 2]
    # x1, x2: [#cluster_points, 2] where x1 and x2 belongs in different clusters
    plt.plot(x1[:, 0].numpy(), x1[:, 1].numpy(), "ro")
    plt.plot(x2[:, 0].numpy(), x2[:, 1].numpy(), "bo")
    l = plt.plot(c[:, 0].numpy(), c[:, 1].numpy(), "kx")
    plt.setp(l, markersize=10)
    plt.show()
