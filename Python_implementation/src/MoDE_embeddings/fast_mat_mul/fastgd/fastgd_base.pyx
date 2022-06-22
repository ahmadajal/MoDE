import  numpy as np
from scipy import sparse


def gd(inc_mat, N, r_lb, r_ub, gamma, max_iter, verbose, tol):
    # Initialization of the GD algorithm
    # initialize angle values with zero

    x = np.zeros(N - 1)
    # keeping the progress of algorithm
    error_progression = np.zeros(max_iter)
    if verbose:
        print("Start of Gradient Descent algorithm")
    inc_mat_tr = sparse.csr_matrix(inc_mat.T)
    for cnt in range(max_iter):
        Ax = inc_mat.dot(x)
        A_sym_x = inc_mat_tr.dot(Ax)
        proj_x = proj_l_u(Ax, r_lb, r_ub)
        A_diff = A_sym_x - inc_mat_tr.dot(proj_x)

        if cnt % 10000 == 0 and verbose:
            print("{} out of {} iterations has passed".format(cnt, max_iter))
        e = (1 / np.sqrt(N - 1)) * np.linalg.norm(A_diff)
        error_progression[cnt] = e

        # check if the error is below tolerance
        if cnt % 1000 == 0 and e < tol:
            if verbose:
                print("GD stopped after {} iteration".format(cnt))
            error_progression = error_progression[: cnt + 1]
            break  # here the algorithm finishes

        # The update step
        x = x - gamma * (A_diff)

    return x




def proj_l_u(x, l , u):
    return np.minimum(np.maximum(x, l), u)
