import numpy as np
import scipy
from scipy.sparse import identity, find, csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

class MoDE:

    def __init__(self, n_neighbor=20, max_iter=10000, tol=0.001, n_components=2, verbose=False):
        """
        Implementation of the paper "An Interpretable Data Embedding under Uncertain Distance Information"
        <link_to_the_paper>
        This class computes the Multi-objective 2D Embeddings (MoDE) for the input dataset.

        n_neighbor: int, Number of nearest neighbors used to create the data graph. This parameter is similar to
        the number of nearest neighbors used in other manifold learning algorithms (e.g, ISOMAP).
        max_iter: int, Maximum number of iterations for gradient descent to solve the optimization problem
        tol: float, Tolerance value used as a stop condition for the gradient descent algorithm. GD stops either
        if the it reaches the maximum number of iterations or the error becomes smaller than this tolerance value.
        n_components: dimensionality of the output embeddings
        verbose: (Default = False) If true, the progress of the gradient descent algorithm will be printed while
        the embeddings are being computed.
        """
        self.n_neighbor = n_neighbor
        self.max_iter = max_iter
        self.verbose = verbose
        self.n_components = n_components
        self.tol = tol

    def fit_transform(self, data, score, dm_ub=None, dm_lb=None):
        """
        Fit data into an embedded space and return the transformed 2D output

        data: array of shape (n_samples, n_features), i.e, it should contain a sample per row
        score: array of shape (n_samples,) that contain the score (ranking) for each sample. Some datasets have
        ranked data points by nature, e.g, market value of each stock in a dataset of stocks, rank of each university
        in a data set of universities, etc. In case such scores are not available in a dataset, random scores can be
        used
        dm_ub: array of shape (n_samples, n_samples) that contain the upper-bound on the mutual distance of data
        samples from each other. In some cases, like data compression, exact pair-wise distances between data points
        are not available. In such cases ranges of upper and lower bound distances between data points can be computed.
        MoDE can operate on such distance bounds. In the case where exact distance information are available, just pass
        the exact distance matrix to both `dm_ub` and `dm_lb`. If "None" then the exact distance matrix will be computed.
        dm_lb: array of shape (n_samples, n_samples) that contain the lower-bound on the mutual distance of data
        samples from each other. If "None" then the exact distance matrix will be computed.
        :return: x_2d: array of shape (n_samples, 2). Embedding of the training data in 2D space.
        """
        N = data.shape[0]
        if dm_ub is None or dm_lb is None:
            dm = pairwise_distances(data, n_jobs=-1)
            dm = np.round(dm, decimals=5)
            dm_ub = dm
            dm_lb = dm
        # check if distance matrices are symmetric
        if np.any(dm_ub.T != dm_ub) or np.any(dm_lb.T != dm_lb.T):
            raise Exception("distance matrices should be symmetric")
        # compute the norm of each point
        data_norms = np.linalg.norm(data, axis=1)
        if 0 in data_norms:
            raise Exception("error: remove zero-norm points")
        # compute the correlation lower and upper bound
        data_norms_i = np.repeat(data_norms, repeats=N).reshape((N, N)).T
        data_norms_j = np.repeat(data_norms, repeats=N).reshape((N, N))
        cm_ub = (data_norms_i ** 2 + data_norms_j ** 2 - dm_lb ** 2) / (2 * data_norms_i * data_norms_j)
        cm_lb = (data_norms_i ** 2 + data_norms_j ** 2 - dm_ub ** 2) / (2 * data_norms_i * data_norms_j)

        # create the KNN Graph
        # take the average distances to create the KNNG
        dm = (dm_ub + dm_lb) / 2
        # we use n_neighbor+1 in order to exclude a point being nearest neighbor with itself later
        neigh = NearestNeighbors(n_neighbors=self.n_neighbor+1, metric="precomputed", n_jobs=-1)
        neigh.fit(dm)
        # compute the adjacency matrix
        A = neigh.kneighbors_graph(dm) - identity(N, format="csr")
        # construct the incidence matrix
        inc_mat = self.incidence_matrix(A, score)
        # Bounds on correlation (vectors of length = # edges)
        node_indices = inc_mat.nonzero()[1].reshape((-1, 2))
        c_ub = cm_ub[node_indices[:, 0], node_indices[:, 1]]
        c_lb = cm_lb[node_indices[:, 0], node_indices[:, 1]]
        # first we find the index of the point with the lowest score and remove it from incidence matrix
        min_ind = np.argmin(score.squeeze())
        inc_mat = inc_mat[:, list(range(min_ind)) + list(range(min_ind+1, N))]
        # we keep a matrix P containing all the angles, size: N * (n_components-1)
        self.P = np.zeros((N, self.n_components-1))*0.01 # was initialized by zero before
        for phi in range(self.n_components-1):
            # Bounds on angular difference.
            # note that acos() is a decreasing function
            if phi == 0:
                r_ub = np.arccos(c_lb)
                r_lb = np.arccos(c_ub)
            else:
                if phi == 1:  #correlation in p=2 dimensions
                    x_pd_norms = np.linalg.norm(x_pd, axis=1)
                    c_p = np.einsum("ij,ij->i", x_pd[node_indices[:, 0]], x_pd[node_indices[:, 1]]) / \
                    (x_pd_norms[node_indices[:, 0]] * x_pd_norms[node_indices[:, 1]])
                else: #correlation in p>2 dimensions, eq 8 in the paper
                    xx = np.prod(np.sin(self.P[node_indices[:, 0], :phi-1]), axis=1) * \
                            np.prod(np.sin(self.P[node_indices[:, 1], :phi-1]), axis=1) * \
                            (np.cos(self.P[node_indices[:, 1], phi-1] - self.P[node_indices[:, 0], phi-1]) - 1)
                    c_p = c_p + xx
                    print((xx<=0).all())
                denom = np.prod(np.sin(self.P[node_indices[:, 0], :phi]), axis=1) * \
                        np.prod(np.sin(self.P[node_indices[:, 1], :phi]), axis=1)
                r_ub = np.arccos(self.proj_l_u(1 + (c_lb - c_p) / denom, -1, 1))
                r_lb = np.arccos(self.proj_l_u(1 + (c_ub - c_p) / denom, -1, 1))
            # Initialization of the GD algorithm
            # initialize angle values with zero
            x = np.zeros(N-1)
            # keeping the progress of algorithm
            error_progression = np.zeros(self.max_iter)
            gamma = 1 / (2 * np.max((np.dot(inc_mat.T, inc_mat)).diagonal()))
            print(gamma)
            if self.verbose:
                print("Start of Gradient Descent algorithm")
            for cnt in range(self.max_iter):
                if cnt%10000 == 0 and self.verbose:
                    print("{} out of {} iterations has passed".format(cnt, self.max_iter))
                    # print(x)

                e = (1/np.sqrt(N-1)) * np.linalg.norm(inc_mat.T.dot(inc_mat.dot(x) - self.proj_l_u(inc_mat.dot(x), r_lb, r_ub)))
                error_progression[cnt] = e
                # check if the error is below tolerance
                if cnt % 1000 == 0 and e < self.tol:
                    if self.verbose:
                        print("GD stopped after {} iteration".format(cnt))
                    error_progression = error_progression[:cnt+1]
                    break  # here the algorithm finishes
                # The update step
                x = x - gamma * inc_mat.T.dot(inc_mat.dot(x) - self.proj_l_u(inc_mat.dot(x), r_lb, r_ub))
            # adding back the point with the least score
            x = np.concatenate((x[:min_ind], np.array([0.01]), x[min_ind:]), axis=0)
            if self.verbose:
                print("end of GD algorithm")
            # keeping the resulting angles
            self.P[:, phi] = x
            # generating the points in phi+1 dimensions
            if phi == 0:
                x_pd = np.concatenate((data_norms * np.cos(x), data_norms * np.sin(x)), axis=0).reshape((2, -1)).T
            else:
                x_pd = self.to_hyper_spherical(data_norms, self.P[:, :phi+1])
        return x_pd

    def incidence_matrix(self, A, score):
        """
        Creates the sparse incidence matrix of a graph from its adjacency matrix. More information about incidence
        matrix could be found in the paper

        A: array of shape (n_nodes, n_nodes) Graph adjacency matrix (created from the k-nearest neighbors
        data graph). Here n_nodes = n_samples.
        score: Score (ranking) value for each data point
        :return: inc_mat: array of shape (n_edges, n_nodes), sparse incidence matrix of the graph
        """
        (m, n) = A.shape
        if m != n:
            raise Exception("error: adjacency matrix should be a square matrix")
        if np.any((find(A)[2] != 1) & (find(A)[2] != 0)):
            raise ValueError("not a 0-1 matrix")
        if len(score) != m:
            raise Exception("error: length of the score vector should be equal to the number of data points")
        # create the set of edges of the KNN graph, with nodes sorted according to score, i.e, (i, j) for i < j
        edges = set([tuple(sorted(x, key=lambda y: score[y])) for x in zip(find(A)[0], find(A)[1])])
        # temporary:
        # edges = []
        # for t in zip(find(A.T)[1], find(A.T)[0]):
        #     if tuple(sorted(t)) not in edges:
        #         edges.append(t)
        # edges = [tuple(sorted(x, key=lambda y: score[y])) for x in edges]

        row_ind = []
        col_ind = []
        values = []
        for i, e in enumerate(edges):
            row_ind = row_ind + [i, i]
            col_ind = col_ind + list(e)
            values = values + [-1, 1]
        inc_mat = csr_matrix((values, (row_ind, col_ind)))
        return inc_mat

    def proj_l_u(self, x, l, u):
        """
        project the values of an array into the bound [l, u] (element-wise)

        x: input array
        l: array of lower bounds
        u: array of upper bounds
        :return: projected output array
        """
        return np.minimum(np.maximum(x, l), u)

    def to_hyper_spherical(self, r, angles):
        """
        convert array x from cartesian to hyper-spherical coordinates

        r: norm of the data points (N * 1 vector)
        angles: angles of the hyper_spherical coordinates (N * p-1 matrix)
        :return: output data in cartesian coordinates (N * p matrix)
        """
        r = np.array(r)
        N = len(r)
        angles = np.array(angles)
        if angles.shape[0] != N:
            raise ValueError("dimension of the norms and angles array do not match: {}, {}".format(r.shape, angles.shape))
        x_cart = np.zeros((N, angles.shape[1] + 1))
        for i in range(angles.shape[1]):
            x_cart[:, i] = r * np.prod(np.sin(angles[:, :i]), axis=1) * np.cos(angles[:, i])
        x_cart[:, -1] = r * np.prod(np.sin(angles), axis=1)
        return x_cart
