import  numpy as np
from scipy import sparse
cimport cython
from cpython cimport bool
from time import time


@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def gd(inc_mat,long N, double[::1] r_lb,double[::1] r_ub, float gamma, long max_iter, bool verbose, float tol):
    # Initialization of the GD algorithm
    # initialize angle values with zero

    x_arr = np.zeros(N - 1, dtype = np.double)
    cdef double[::1] x = np.zeros(N - 1, dtype = np.double)

    # keeping the progress of algorithm
    error_progression_arr = np.zeros(max_iter, dtype = np.double)
    cdef double[::1] error_progression = error_progression_arr

    if verbose:
        print("Start of Gradient Descent algorithm")

    inc_mat_tr = sparse.csr_matrix(inc_mat.T)

    cdef Py_ssize_t cnt


    cdef double[::1] Ax
    cdef double[::1] A_sym_x
    cdef double[::1] proj_x
    cdef double[::1] A_diff
    cdef double e

    for cnt in range(max_iter):
        #t1 = time()
        Ax = inc_mat.dot(x)
        #t2 = time()
        A_sym_x = inc_mat_tr.dot(Ax)
        #t3 = time()
        proj_x = proj_l_u(Ax, r_lb, r_ub)
        #t4 = time()
        A_diff = A_sym_x - inc_mat_tr.dot(proj_x)
        if cnt % 10000 == 0 and verbose:
            print("{} out of {} iterations has passed".format(cnt, max_iter))
        #t5 = time()

        e = (1/np.sqrt(N - 1)) * np.linalg.norm(A_diff)
        error_progression[cnt] = e

        # check if the error is below tolerance
        if cnt % 1000 == 0 and e < tol:
            if verbose:
                print("GD stopped after {} iteration".format(cnt))
            error_progression = error_progression[: cnt + 1]
            break  # here the algorithm finishes

        # The update step
        x = x - gamma * np.asarray(A_diff)
        #t6 = time()
        #print("rest time : {}".format((t5 - t4)/(t6-t1)))


    return x



@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)

def proj_l_u(double[::1] x, double[::1] l ,double[::1] u):
    return np.maximum(np.minimum(x,u),l)

