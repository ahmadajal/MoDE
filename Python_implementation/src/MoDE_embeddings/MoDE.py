import itertools

import numpy as np
import scipy
from scipy.sparse import csr_matrix, find, identity
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from .fastgd import fastgd_base, fastgd_cython, fastgd_faster


class MoDE:
    def __init__(
        self,
        n_neighbor: int = 20,
        max_iter: int = 10000,
        tol: float = 0.001,
        n_components: int = 2,
        verbose: bool = False,
        method: str = "fastgd_faster",
    ):
        """Implementation of the paper "Interpretable embedding and visualization
        of compressed data" https://dl.acm.org/doi/full/10.1145/3537901
        This class computes the Multi-objective Data Embeddings (MoDE) for the input dataset.

        Args:
            n_neighbor (int, optional): int, Number of nearest neighbors used to create the
            data graph.
            This parameter is similar to the number of nearest neighbors used in other
            manifold learning algorithms (e.g, ISOMAP). Defaults to 20.
            max_iter (int, optional):  Maximum number of iterations for gradient descent to s
            olve the optimization problem. Defaults to 10000.
            tol (float, optional): Tolerance value used as a stop condition for the gradient
            descent algorithm. GD stops either if the it reaches the maximum number of iterations
            or the error becomes smaller than this tolerance value. Defaults to 0.001.
            n_components (int, optional): dimensionality of the output embeddings.
            Defaults to 2.
            verbose (bool, optional): the progress of the gradient descent algorithm will be
            printed while the embeddings are being computed. Defaults to False.
            method (str, optional): Selects the gradient decsent implementation to be used.
            Defaults to "fastgd_faster".
        """
        self.n_neighbor = n_neighbor
        self.max_iter = max_iter
        self.verbose = verbose
        self.n_components = n_components
        self.tol = tol
        self.method = method

    def fit_transform(
        self,
        data: np.ndarray,
        score: np.array,
        dm_ub: np.ndarray = None,
        dm_lb: np.ndarray = None,
    ) -> np.ndarray:
        """Fit data into an embedded space and return the transformed output in lower dimensions.


        :return: x_2d: array of shape (n_samples, 2). Embedding of the training data in 2D space.


        Args:
            data (np.ndarray): array of shape (n_samples, n_features), i.e, it should contain a
            sample per row.
            score (np.array): array of shape (n_samples,) that contain the score (ranking) for
            each sample. Some datasets have ranked data points by nature, e.g, market value of
            each stock in a dataset of stocks, rank of each university in a data set of
            universities, etc. In case such scores are not available in a dataset, random scores
            can be used.
            dm_ub (np.ndarray, optional): array of shape (n_samples, n_samples) that contain the
            upper-bounds on the pairwise distance of data points. Defaults to None.
            dm_lb (np.ndarray, optional): array of shape (n_samples, n_samples) that contain the
            lower-bounds on the pairwise distance of data points. Defaults to None.


        Returns:
            np.ndarray: array of shape (n_samples, self.n_components). Embedding of the training
            data in low-dimensional space.
        """
        print("start")
        N = data.shape[0]
        # compute the norm of each point
        if isinstance(data, csr_matrix):
            data_norms = scipy.sparse.linalg.norm(data, axis=1)
        else:
            data_norms = np.linalg.norm(data, axis=1)
        if 0 in data_norms:
            raise Exception("error: remove zero-norm points")
        if dm_ub is None or dm_lb is None:
            # we use n_neighbor+1 in order to exclude a point being nearest neighbor with
            # itself later.
            neigh = NearestNeighbors(n_neighbors=self.n_neighbor + 1, n_jobs=-1)
            neigh.fit(data)
            # compute the adjacency matrix
            A = neigh.kneighbors_graph(data) - identity(N, format="csr")
        else:
            # check if distance matrices are symmetric
            if np.any(dm_ub.T != dm_ub) or np.any(dm_lb.T != dm_lb.T):
                raise Exception("distance matrices should be symmetric")
            # take the average distances to create the KNNG
            dm = (dm_ub + dm_lb) / 2
            # we use n_neighbor+1 in order to exclude a point being nearest neighbor with
            # itself later.
            neigh = NearestNeighbors(
                n_neighbors=self.n_neighbor + 1, metric="precomputed", n_jobs=-1
            )
            neigh.fit(dm)
            # compute the adjacency matrix
            A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
        print("KNN graph computed.")
        # construct the incidence matrix
        inc_mat = self.incidence_matrix(A, score)
        # Compute the bounds on correlation (vectors of length = # edges)
        node_indices = inc_mat.nonzero()[1].reshape((-1, 2))
        c_ub = np.zeros(len(node_indices))
        c_lb = np.zeros(len(node_indices))
        for i, ind in enumerate(node_indices):
            if dm_ub is None or dm_lb is None:
                if isinstance(data, csr_matrix):
                    d_lb = scipy.sparse.linalg.norm(data[ind[0]] - data[ind[1]])
                else:
                    d_lb = distance.euclidean(data[ind[0]], data[ind[1]])
                d_ub = d_lb
            else:
                d_lb = dm_lb[ind[0], ind[1]]
                d_ub = dm_ub[ind[0], ind[1]]
            c_ub[i] = (data_norms[ind[0]] ** 2 + data_norms[ind[1]] ** 2 - d_lb**2) / (
                2 * data_norms[ind[0]] * data_norms[ind[1]]
            )
            c_lb[i] = (data_norms[ind[0]] ** 2 + data_norms[ind[1]] ** 2 - d_ub**2) / (
                2 * data_norms[ind[0]] * data_norms[ind[1]]
            )
        print("Bounds on correlations computed.")
        # first we find the index of the point with the lowest score and remove it from
        # incidence matrix.
        min_ind = np.argmin(score.squeeze())
        inc_mat = inc_mat[:, list(range(min_ind)) + list(range(min_ind + 1, N))]
        # we keep a matrix P containing all the angles, size: N * (n_components-1)
        self.P = np.zeros((N, self.n_components - 1)) * 0.01  # was initialized by zero before
        for phi in range(self.n_components - 1):
            # Bounds on angular difference.
            # note that acos() is a decreasing function
            if phi == 0:
                r_ub = np.arccos(c_lb)
                r_lb = np.arccos(c_ub)
            else:
                if phi == 1:  # correlation in p=2 dimensions
                    x_pd_norms = np.linalg.norm(x_pd, axis=1)  # noqa: F821
                    c_p = np.einsum(
                        "ij,ij->i", x_pd[node_indices[:, 0]], x_pd[node_indices[:, 1]]  # noqa: F821
                    ) / (x_pd_norms[node_indices[:, 0]] * x_pd_norms[node_indices[:, 1]])
                else:  # correlation in p>2 dimensions, eq 8 in the paper
                    xx = (
                        np.prod(np.sin(self.P[node_indices[:, 0], : phi - 1]), axis=1)
                        * np.prod(np.sin(self.P[node_indices[:, 1], : phi - 1]), axis=1)
                        * (
                            np.cos(
                                self.P[node_indices[:, 1], phi - 1]
                                - self.P[node_indices[:, 0], phi - 1]
                            )
                            - 1
                        )
                    )
                    c_p = c_p + xx
                    print((xx <= 0).all())
                denom = np.prod(np.sin(self.P[node_indices[:, 0], :phi]), axis=1) * np.prod(
                    np.sin(self.P[node_indices[:, 1], :phi]), axis=1
                )
                r_ub = np.arccos(self.proj_l_u(1 + (c_lb - c_p) / denom, -1, 1))
                r_lb = np.arccos(self.proj_l_u(1 + (c_ub - c_p) / denom, -1, 1))

            gamma = 1 / (2 * np.max((np.dot(inc_mat.T, inc_mat)).diagonal()))

            if self.method == "base":
                x = self.gd_iter(inc_mat, N, r_lb, r_ub, gamma)
            elif self.method == "fastgd_base":
                x = fastgd_base.gd(
                    inc_mat, N, r_lb, r_ub, gamma, self.max_iter, self.verbose, self.tol
                )
            elif self.method == "fastgd_cython":
                x = fastgd_cython.gd(
                    inc_mat, N, r_lb, r_ub, gamma, self.max_iter, self.verbose, self.tol
                )
            elif self.method == "fastgd_faster":
                x = fastgd_faster.gd(
                    inc_mat, N, r_lb, r_ub, gamma, self.max_iter, self.verbose, self.tol
                )

            # adding back the point with the least score
            x = np.concatenate((x[:min_ind], np.array([0.01]), x[min_ind:]), axis=0)

            if self.verbose:
                print("end of GD algorithm")
            # keeping the resulting angles
            self.P[:, phi] = x
            # generating the points in phi+1 dimensions
            if phi == 0:
                x_pd = (
                    np.concatenate((data_norms * np.cos(x), data_norms * np.sin(x)), axis=0)
                    .reshape((2, -1))
                    .T
                )
            else:
                x_pd = self.from_hyper_spherical(data_norms, self.P[:, : phi + 1])
        return x_pd

    def incidence_matrix(self, A: np.ndarray, score: np.array) -> csr_matrix:
        """Creates the sparse incidence matrix of a graph from its adjacency matrix.
        More information about incidence matrix could be found in the paper.

        A: array of shape (n_nodes, n_nodes) Graph adjacency matrix (created from
        the k-nearest neighbors data graph). Here n_nodes = n_samples.
        score: Score (ranking) value for each data point
        :return: inc_mat: array of shape (n_edges, n_nodes), sparse incidence matrix of the graph


        Args:
            A (np.ndarray): Array of shape (n_nodes, n_nodes). Graph adjacency matrix, created from
            the k-nearest neighbors data graph. Here n_nodes = n_samples.
            score (np.array): Score (ranking) value for each data point.

        Returns:
            csr_matrix: Sparse incidence matrix of the graph with the shape (n_edges, n_nodes).
        """
        (m, n) = A.shape
        if m != n:
            raise Exception("error: adjacency matrix should be a square matrix")
        if np.any((find(A)[2] != 1) & (find(A)[2] != 0)):
            raise ValueError("not a 0-1 matrix")
        if len(score) != m:
            raise Exception(
                "error: length of the score vector should be equal to the number of data points"
            )
        # create the set of edges of the KNN graph, with nodes sorted according to score,
        # i.e, (i, j) for i < j.
        edges = set([tuple(sorted(x, key=lambda y: score[y])) for x in zip(find(A)[0], find(A)[1])])
        # t2 = time.time()
        print("Incidence matrix created ")

        row_ind = np.repeat(range(len(edges)), 2)
        col_ind = list(itertools.chain(*edges))
        values = [-1, 1] * len(edges)
        inc_mat = csr_matrix((values, (row_ind, col_ind)))
        return inc_mat

    def proj_l_u(self, x: np.array, l: np.array, u: np.array) -> np.array:  # noqa: E741
        """project the values of an array into the bound [l, u] (element-wise).


        Args:
            x (np.array): Input array.
            l (np.array): Array of lower bounds
            u (np.array): Array of upper bounds

        Returns:
            np.array: Projected output array.
        """
        return np.minimum(np.maximum(x, l), u)

    def from_hyper_spherical(self, r: np.array, angles: np.ndarray) -> np.ndarray:
        """convert array x from hyper-spherical to cartesian coordinates.

        r: norm of the data points (N * 1 vector)
        angles: angles of the hyper_spherical coordinates (N * p-1 matrix)
        :return: output data in cartesian coordinates (N * p matrix)

        Args:
            r (np.array): Norm of the data points.
            angles (np.ndarray): Angles of the hyper_spherical coordinates
            (N * p-1 matrix for p dimensions).

        Returns:
            np.ndarray: Output data in cartesian coordinates (array of shape N * p).

        """
        r = np.array(r)
        N = len(r)
        angles = np.array(angles)
        if angles.shape[0] != N:
            raise ValueError(
                "dimension of the norms and angles array do not match: {}, {}".format(
                    r.shape, angles.shape
                )
            )
        x_cart = np.zeros((N, angles.shape[1] + 1))
        for i in range(angles.shape[1]):
            x_cart[:, i] = r * np.prod(np.sin(angles[:, :i]), axis=1) * np.cos(angles[:, i])
        x_cart[:, -1] = r * np.prod(np.sin(angles), axis=1)
        return x_cart

    def gd_iter(
        self, inc_mat: csr_matrix, N: int, r_lb: np.array, r_ub: np.array, gamma: float
    ) -> np.ndarray:
        """Python (slow) implementaion of the Gradient Decsent algorithm (Deprecated).

        Args:
            inc_mat (csr_matrix): Sparse incidence matrix of the graph with the shape
            (n_edges, n_nodes).
            N (int): Number of data points.
            r_lb (np.array): Lower bound on angular values.
            r_ub (np.array): Upper bound on angular values.
            gamma (float): Stepsize of the GD algorithm.

        Returns:
            np.ndarray: Array of the optimized angular values.
        """

        x = np.zeros(N - 1)
        # keeping the progress of algorithm
        error_progression = np.zeros(self.max_iter)
        if self.verbose:
            print("Start of Gradient Descent algorithm")
        inc_mat_tr = csr_matrix(inc_mat.T)
        for cnt in range(self.max_iter):
            Ax = inc_mat.dot(x)
            proj_x = self.proj_l_u(Ax, r_lb, r_ub)
            A_diff = inc_mat_tr.dot(Ax - proj_x)

            if cnt % 10000 == 0 and self.verbose:
                print("{} out of {} iterations has passed".format(cnt, self.max_iter))

            e = (1 / np.sqrt(N - 1)) * np.linalg.norm(A_diff)
            error_progression[cnt] = e
            # check if the error is below tolerance
            if cnt % 1000 == 0 and e < self.tol:
                if self.verbose:
                    print("GD stopped after {} iteration".format(cnt))
                error_progression = error_progression[: cnt + 1]
                break  # here the algorithm finishes
            # The update step
            # first_dot = inc_mat.dot(x)
            x = x - gamma * (A_diff)

        return x
