import numpy as np
import scipy
from scipy.sparse import identity, find, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def distance_metric(data, x_2d, n_neighbor, dm=None):
    """
    Compute the distance preservation metric for the embedded data. The range of the output values are between 0 and 1,
    with larger values showing that the projected dataset in the embedding space has a higher fidelity in preserving
    pair-wise distances. This metric considers the preservation of pair-wise distances only among the `n_neighbor`
    nearest neighbors of each data point. More information on this metric can be found in the paper: "An Interpretable
    Data Embedding under Uncertain Distance Information"

    data: array of shape (n_samples, n_features), input dataset
    x_2d: array of shape (n_samples, dim_embedding_space), projected dataset in the embedding space
    n_neighbor: int, number of nearest neighbors used for computing the embeddings
    dm: array of shape (n_samples, n_samples), average of upper and lower bound distance matrices should be
    given in order to create the same KNNG used in training MoDE embeddings. In case exact distance matrix was used to
    train MoDE embeddings, you should pass the exact distance matrix to this attribute. If "None" the exact distance matrix will be used.
    :return: R_d, distance preservation metric value
    """
    N = data.shape[0]
    if dm is None:
        # creating the adjacency matrix for KNNG using the data.
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, n_jobs=-1)
        neigh.fit(data)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(data) - identity(N, format="csr")
    else:
        # creating the adjacency matrix for KNNG using the provided distance matrix
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, metric="precomputed", n_jobs=-1)
        neigh.fit(dm)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
    edges = set([tuple(sorted(x)) for x in zip(find(A)[0], find(A)[1])])
    # cost of distance preservation for each pair
    if isinstance(data, csr_matrix):
        c = [abs(scipy.sparse.linalg.norm(data[e[0]] - data[e[1]]) - np.linalg.norm(x_2d[e[0]] - x_2d[e[1]])) /
        (scipy.sparse.linalg.norm(data[e[0]] - data[e[1]]) + np.linalg.norm(x_2d[e[0]] - x_2d[e[1]])) for e in edges]
    else:
        c = [abs(np.linalg.norm(data[e[0]] - data[e[1]]) - np.linalg.norm(x_2d[e[0]] - x_2d[e[1]])) /
        (np.linalg.norm(data[e[0]] - data[e[1]]) + np.linalg.norm(x_2d[e[0]] - x_2d[e[1]])) for e in edges]
    R_d = 1 - np.mean(c)
    return R_d


def correlation_metric(data, x_2d, n_neighbor, dm=None):
    """
    Compute the correlation preservation metric for the embedded data. The range of the output values are between -1 and 1,
    with larger values showing that the projected dataset in the embedding space has a higher fidelity in preserving
    pair-wise correlations. This metric considers the preservation of pair-wise correlations only among the `n_neighbor`
    nearest neighbors of each data point. More information on this metric can be found in the paper: "An Interpretable
    Data Embedding under Uncertain Distance Information"

    data: array of shape (n_samples, n_features), input dataset
    x_2d: array of shape (n_samples, dim_embedding_space), projected dataset in the embedding space
    n_neighbor: int, number of nearest neighbors used for computing the embeddings
    dm: array of shape (n_samples, n_samples), average of upper and lower bound distance matrices should be
    given in order to create the same KNNG used in training MoDE embeddings. In case exact distance matrix was used to
    train MoDE embeddings, you should pass the exact distance matrix to this attribute. If "None" the exact distance matrix will be used.
    :return: R_c, correlation preservation metric value
    """
    N = data.shape[0]
    if dm is None:
        # creating the adjacency matrix for KNNG using the data.
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, n_jobs=-1)
        neigh.fit(data)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(data) - identity(N, format="csr")
    else:
        # creating the adjacency matrix for KNNG using the provided distance matrix
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, metric="precomputed", n_jobs=-1)
        neigh.fit(dm)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
    edges = set([tuple(sorted(x)) for x in zip(find(A)[0], find(A)[1])])
    # original correlations
    if isinstance(data, csr_matrix):
        corr_orig = [data[e[0]].dot(data[e[1]].T).data[0] / (scipy.sparse.linalg.norm(data[e[0]]) * scipy.sparse.linalg.norm(data[e[1]])) for e in edges]
    else:
        corr_orig = [np.inner(data[e[0]], data[e[1]]) / (np.linalg.norm(data[e[0]]) * np.linalg.norm(data[e[1]])) for e in edges]
    # embedded data correlations
    corr_emb = [np.inner(x_2d[e[0]], x_2d[e[1]]) / (np.linalg.norm(x_2d[e[0]]) * np.linalg.norm(x_2d[e[1]])) for e in edges]
    # cost of correlation preservation for each pair
    c = np.abs(np.array(corr_orig) - np.array(corr_emb))
    R_c = 1 - np.mean(c)
    return R_c


def order_preservation(data, angles, n_neighbor, score, dm=None):
    """
    Compute the order preservation metric for the embedded data. The range of the output values are between 0 and 1,
    with larger values showing that the projected dataset in the embedding space has a higher fidelity in preserving
    orders of the data points (data points with higher ranks are places in higher angles in 2D space). This metric
    considers the preservation of orders only among the `n_neighbor` nearest neighbors of each data point. This metric
    should be used only for MoDE embeddings.
    More information on this metric can be found in the paper: "An Interpretable
    Data Embedding under Uncertain Distance Information"

    data: array of shape (n_samples, n_features), input dataset
    angles: The array of angles for which you want to compute order preservation. Note that for MoDE Embeddings
    in p>2 dimensions you have a matrix of angles of size N \times p1 and you can compute order order_preservation
    for each column of this matrix.
    n_neighbor: int, number of nearest neighbors used for computing the embeddings
    score: Score (ranking) value for each data point
    dm: array of shape (n_samples, n_samples), average of upper and lower bound distance matrices should be
    given in order to create the same KNNG used in training MoDE embeddings. In case exact distance matrix was used to
    train MoDE embeddings, you should pass the exact distance matrix to this attribute. If "None" then exact distance matrix will be used.
    :return: R_o, order preservation metric value
    """
    N = data.shape[0]
    if dm is None:
        # creating the adjacency matrix for KNNG using the data.
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, n_jobs=-1)
        neigh.fit(data)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(data) - identity(N, format="csr")
    else:
        # creating the adjacency matrix for KNNG using the provided distance matrix
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=n_neighbor + 1, metric="precomputed", n_jobs=-1)
        neigh.fit(dm)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
    edges = set([tuple(sorted(x)) for x in zip(find(A)[0], find(A)[1])])
    # cost of order preservation for each pair
    c = [order_check(angles[e[0]], angles[e[1]], score[e[0]], score[e[1]]) for e in edges]
    R_o = 1 - np.mean(c)
    return R_o

def order_check(theta1, theta2, score_x1, score_x2):
    """
    check if two data points in the 2D embedded space are placed in the correct order. Data points with higher score
    should be placed in higher angles in polar coordinates (for MoDE embeddings).
    x1: array, first data point in the 2D embedded space
    x2: array, second data point in the 2D embedded space
    score_x1: int, score of the first data point
    score_x2: int, score of the second data point
    :return: 1 if the order is preserved, 0 otherwise
    """
    # check if the order is not preserved
    if ((score_x1 < score_x2) & (theta1 > theta2)) | ((score_x1 > score_x2) & (theta1 < theta2)):
        # print(theta1, score_x1, theta2, score_x2)
        return 1
    else:
        return 0


def cart2pol(x, y):
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta


def order_check_old(x1, x2, score_x1, score_x2):
    """
    check if two data points in the 2D embedded space are placed in the correct order. Data points with higher score
    should be placed in higher angles in polar coordinates (for MoDE embeddings).
    x1: array, first data point in the 2D embedded space
    x2: array, second data point in the 2D embedded space
    score_x1: int, score of the first data point
    score_x2: int, score of the second data point
    :return: 1 if the order is preserved, 0 otherwise
    """
    _, theta1 = cart2pol(x1[0], x1[1])
    _, theta2 = cart2pol(x2[0], x2[1])
    # check if the order is not preserved
    if ((score_x1 < score_x2) & (theta1 > theta2)) | ((score_x1 > score_x2) & (theta1 < theta2)):
        # print(theta1, score_x1, theta2, score_x2)
        return 1
    else:
        return 0
