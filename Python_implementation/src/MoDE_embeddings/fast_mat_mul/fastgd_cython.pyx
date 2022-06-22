import  numpy as np
from scipy import sparse
cimport cython

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def gd(inc_mat,long N, double[:] r_lb,double[:] r_ub, float gamma, long max_iter, bool verbose, float tol):
    # Initialization of the GD algorithm
    # initialize angle values with zero

    x_arr = np.zeros(N - 1, dtype = np.double)
    cdef double[:] x = np.zeros(N - 1, dtype = np.double)

    # keeping the progress of algorithm
    error_progression_arr = np.zeros(max_iter, dtype = np.double)
    cdef double[:] error_progression = error_progression_arr

    if verbose:
        print("Start of Gradient Descent algorithm")

    inc_mat_tr = sparse.csr_matrix(inc_mat.T)

    gamma_arr = gamma * np.ones(N - 1, dtype=np.double)
    cdef double[:] gamma_view = gamma_arr

    cdef Py_ssize_t cnt


    cdef double[:] Ax
    cdef double[:] A_sym_x
    cdef double[:] proj_x
    cdef double[:] A_diff
    cdef double e

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
        x = x - gamma_view * (A_diff)

    return x




def proj_l_u(double[:] x, double[:] l ,double[:] u):
    cdef double[:] temp
    temp = np.maximum(x,l)
    temp = np.minimum(temp, u)
    return temp
