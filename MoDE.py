import numpy as np
import scipy
from scipy.sparse import identity, find, csr_matrix
from sklearn.neighbors import NearestNeighbors


class MoDE:

    def __init__(self, n_neighbor, max_iter, tol, verbose=False):
        self.n_neighbor = n_neighbor
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol

    def fit_transform(self, data, score, dm_ub, dm_lb):
        N = data.shape[0]
        # check if distance matrices are symmetric
        if np.any(dm_ub.T != dm_ub) or np.any(dm_lb.T != dm_lb.T):
            raise Exception("distance matrices should be symmetric")
        # compute the norm of each point
        data_norms = [np.linalg.norm(data[i]) for i in range(N)]
        # compute the correlation lower and upper bound
        cm_ub = np.eye(N)
        cm_lb = np.eye(N)
        for i in range(N):
            for j in range(i+1, N):
                if data_norms[i] * data_norms[j] == 0:
                    raise Exception("error: remove zero-norm points")
                cm_ub[i, j] = (data_norms[i] ** 2 + data_norms[j] ** 2 - dm_lb[i, j] ** 2) / (2 * data_norms[i] * data_norms[j])
                cm_lb[i, j] = (data_norms[i] ** 2 + data_norms[j] ** 2 - dm_ub[i, j] ** 2) / (2 * data_norms[i] * data_norms[j])

        # make the correlation matrix symmetric
        cm_ub = cm_ub.T + cm_ub - np.eye(N)
        cm_lb = cm_lb.T + cm_lb - np.eye(N)

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
        c_ub = np.array(
            [cm_ub[inc_mat[i].nonzero()[1][0], inc_mat[i].nonzero()[1][1]] for i in range(inc_mat.shape[0])])
        c_lb = np.array(
            [cm_lb[inc_mat[i].nonzero()[1][0], inc_mat[i].nonzero()[1][1]] for i in range(inc_mat.shape[0])])
        # Bounds on angular difference.
        # note that acos() is a decreasing function
        r_ub = np.arccos(c_lb)
        r_lb = np.arccos(c_ub)
        # create a columnar matrix of angles
        y_angle = np.concatenate((r_lb, r_ub), axis=0).T
        # Initialization of the GD algorithm
        # first we find the index of the point with the lowest score and remove it from incidence matrix
        min_ind = np.argmin(score.squeeze())
        inc_mat = inc_mat[:, list(range(min_ind)) + list(range(min_ind+1, N))]
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
        x = np.concatenate((x[:min_ind], np.array([0]), x[min_ind:]), axis=0)
        if self.verbose:
            print("end of GD algorithm")
        # generating the points in 2D
        x_2d = np.concatenate((data_norms * np.cos(x), data_norms * np.sin(x)), axis=0).reshape((2,-1)).T
        return x_2d

    def incidence_matrix(self, A, score):
        (m, n) = A.shape
        if m != n:
            raise Exception("error: adjacency matrix should be a square matrix")
        if np.any((find(A)[2] != 1) & (find(A)[2] != 0)):
            raise ValueError("not a 0-1 matrix")
        if len(score) != m:
            raise Exception("error: length of the score vector should be equal to the number of data points")
        # create the set of edges of the KNN graph, with nodes sorted according to score, i.e, (i, j) for i < j
        # edges = set([tuple(sorted(x, key=lambda y: score[y], reverse=True)) for x in zip(find(A)[0], find(A)[1])])
        # temporary:
        edges = []
        for t in zip(find(A.T)[1], find(A.T)[0]):
            if tuple(sorted(t)) not in edges:
                edges.append(t)
        edges = [tuple(sorted(x, key=lambda y: score[y])) for x in edges]

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
        return np.minimum(np.maximum(x, l), u)






