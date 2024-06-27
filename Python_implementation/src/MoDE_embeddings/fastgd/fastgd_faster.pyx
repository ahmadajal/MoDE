# cython : boundscheck = False
# cython : wraparound = False
# cython : cdivision = True



import  numpy as np
cimport numpy as np
from cpython cimport bool
from libc.stdio cimport printf
from libc.math cimport sqrt, log
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel
from scipy import sparse

np.import_array()



cdef int mat_mult(long[:] m_data, int[:] m_indices, int[:] m_indptr, double* vec, double* res):
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in prange(m_indptr.shape[0]-1, nogil=True):
        res[i] = 0
        for j in range(m_indptr[i], m_indptr[i+1]):
            res[i] += m_data[j] * vec[m_indices[j]]
    return 1




cdef int proj_l_u(double* vec, double[:] l, double[:] u):
    cdef double temp
    for i in range(l.shape[0]):
        temp = min(vec[i], u[i])
        vec[i] -= max(temp, l[i])
    return 1


cdef int comp_grad(long[:] inc_mat_data,
                int[:] inc_mat_indices,
                int[:] inc_mat_indptr,
                long[:] inc_mat_tr_data,
                int[:] inc_mat_tr_indices,
                int[:] inc_mat_tr_indptr,
                double* x,
                double[:] r_lb,
                double[:] r_ub,
                double* grad):

    cdef long N = inc_mat_indptr.shape[0] - 1
    cdef double* Ax = <double*> malloc(sizeof(double)* N)
    mat_mult(inc_mat_data, inc_mat_indices, inc_mat_indptr, x, Ax)
    proj_l_u(Ax, r_lb, r_ub)
    mat_mult(inc_mat_tr_data, inc_mat_tr_indices, inc_mat_tr_indptr, Ax, grad)
    free(Ax)
    return 1
    


cdef double* gd_routine(long[:] inc_mat_data,
                    int[:] inc_mat_indices, 
                    int[:] inc_mat_indptr,
                    long[:] inc_mat_tr_data,
                    int[:] inc_mat_tr_indices,
                    int[:] inc_mat_tr_indptr, 
                    double[:] r_lb, 
                    double[:] r_ub, 
                    double gamma, 
                    long max_iter, 
                    double tol, 
                    bool verbose,
                    long N):

    cdef long cnt
    cdef double* grad = <double*> malloc(sizeof(double)* (N-1))
    cdef double* x = <double*> malloc(sizeof(double) * (N-1))

    cdef long i
    cdef :
        double grad_norm = 0
        double error

    # Initialize x
    for i in range(N-1):
        x[i] = 0

    for cnt in range(max_iter):
        comp_grad(inc_mat_data,
                  inc_mat_indices, 
                  inc_mat_indptr,
                  inc_mat_tr_data,
                  inc_mat_tr_indices,
                  inc_mat_tr_indptr, 
                  x, 
                  r_lb, 
                  r_ub,
                  grad)

        grad_norm = 0
        for i in range(N-1):
            grad_norm += grad[i] * grad[i]

        if grad_norm/(N-1) < tol*tol:
            free(grad)
            return x

        for i in range(N-1):
            x[i] -= gamma * grad[i]
    
    free(grad)
    return x

def gd(inc_mat,long N, double[:] r_lb, double[:] r_ub, double gamma, long max_iter, bool verbose, double tol):

    # Initialization of the GD algorithm
    # initialize angle values with zero

    inc_mat_tr = sparse.csr_matrix(inc_mat.T)
    x = gd_routine(inc_mat.data, inc_mat.indices, inc_mat.indptr, inc_mat_tr.data, inc_mat_tr.indices, inc_mat_tr.indptr, r_lb, r_ub, gamma, max_iter, tol, verbose, N)
    x_arr = np.zeros(N-1, dtype = np.double)
    for i in range(N-1):
        x_arr[i] = x[i]
    print("Number of iterations:")
    return x_arr



