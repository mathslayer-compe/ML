import torch
import hw4_utils


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [N, 2].
        init_c: initial centroids, shape [2, 2]. Each row is a centroid.

    Return:
        c: shape [2, 2]. Each row is a centroid.
    """

    if X is None:
        X, init_c = hw4_utils.load_data()

    k = 2
    n = X.shape[0]

    prev_centroid = init_c.clone()
    for iters in range(n_iters):
        clusters = [torch.zeros((1, 2)) for _ in range(k)]
        for i in range(n):
            closest_cluster = -1
            min_dist = float('inf')
            for j in range(k):
                dist = torch.norm(X[i] - init_c[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = j
            clusters[closest_cluster] = torch.cat((clusters[closest_cluster], X[i].view(1, -1)), 0)

        hw4_utils.vis_cluster(init_c, clusters[0][1:], clusters[1][1:])

        for i in range(k):
            cluster_size = clusters[i].shape[0] - 1 if clusters[i].shape[0] > 1 else 1
            init_c[i] = torch.sum(clusters[i], 0) / cluster_size

        converged = True
        for i in range(k):
            if not torch.equal(prev_centroid[i], init_c[i]):
                converged = False
                break

        if converged:
            cost = sum(torch.norm(clusters[i][1:] - init_c[i]) for i in range(k))
            break

        prev_centroid = init_c.clone()

    return init_c


k_means()