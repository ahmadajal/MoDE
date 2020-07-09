import numpy as np
import scipy
from scipy.sparse import identity, find, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def distance_metric(data, x_2d, dm, n_neighbor):
    """

    :param data:
    :param x_2d:
    :param dm:
    :param n_neighbor:
    :return:
    """
    N = data.shape[0]
    # dm is in general the average of dm_ub and dm_lb and hence could be potentially different from the original distance matrix
    dm_orig = pairwise_distances(data, n_jobs=-1)
    # creating the adjacency matrix for KNNG (from the average distance matrix "dm")
    # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
    neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, metric="precomputed", n_jobs=-1)
    neigh.fit(dm)
    # compute the adjacency matrix
    A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
    edges = set([tuple(sorted(x)) for x in zip(find(A)[0], find(A)[1])])
    # cost of distance preservation for each pair
    c = [abs(dm_orig[e[0], e[1]] - np.linalg.norm(x_2d[e[0]] - x_2d[e[1]])) /
         (dm_orig[e[0], e[1]] + np.linalg.norm(x_2d[e[0]] - x_2d[e[1]])) for e in edges]
    R_d = 1 - np.mean(c)
    return R_d


def correlation_metric(data, x_2d, dm, n_neighbor):
    """

    :param data:
    :param x_2d:
    :param dm:
    :param n_neighbor:
    :return:
    """
    N = data.shape[0]
    # creating the adjacency matrix for KNNG (from the average distance matrix "dm")
    # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
    neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, metric="precomputed", n_jobs=-1)
    neigh.fit(dm)
    # compute the adjacency matrix
    A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
    edges = set([tuple(sorted(x)) for x in zip(find(A)[0], find(A)[1])])
    # original correlations
    corr_orig = [np.inner(data[e[0]], data[e[1]]) / (np.linalg.norm(data[e[0]]) * np.linalg.norm(data[e[1]])) for e in edges]
    # embedded data correlations
    corr_emb = [np.inner(x_2d[e[0]], x_2d[e[1]]) / (np.linalg.norm(x_2d[e[0]]) * np.linalg.norm(x_2d[e[1]])) for e in edges]
    # cost of correlation preservation for each pair
    c = np.abs(np.array(corr_orig) - np.array(corr_emb))
    R_c = 1 - np.mean(c)
    return R_c


def order_preservation(x_2d, dm, n_neighbor, score):
    """

    :param x_2d:
    :param dm:
    :param n_neighbor:
    :param score:
    :return:
    """
    N = x_2d.shape[0]
    # creating the adjacency matrix for KNNG (from the average distance matrix "dm")
    # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
    neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, metric="precomputed", n_jobs=-1)
    neigh.fit(dm)
    # compute the adjacency matrix
    A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
    edges = set([tuple(sorted(x)) for x in zip(find(A)[0], find(A)[1])])
    # cost of order preservation for each pair
    c = [order_check(x_2d[e[0]], x_2d[e[1]], score[e[0]], score[e[1]]) for e in edges]
    R_o = 1 - np.mean(c)
    return R_o


def cart2pol(x, y):
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta


def order_check(x1, x2, score_x1, score_x2):
    _, theta1 = cart2pol(x1[0], x1[1])
    _, theta2 = cart2pol(x2[0], x2[1])
    # check if the order is not preserved
    if ((score_x1 < score_x2) & (theta1 > theta2)) | ((score_x1 > score_x2) & (theta1 < theta2)):
        # print(theta1, score_x1, theta2, score_x2)
        return 1
    else:
        return 0

