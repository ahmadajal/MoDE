import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import identity, find, csr_matrix
from MoDE import MoDE
import scipy

x = np.random.random((5,2))
neigh = NearestNeighbors(n_neighbors=2)
neigh.fit(x)

a = neigh.kneighbors_graph(x, n_neighbors=3) - identity(len(x), format="csr")

print(a.toarray())

score = [1,2,0,-1,3]
print(set([tuple(sorted(x, key=lambda y: score[y], reverse=True)) for x in zip(find(a)[0], find(a)[1])]))

m = MoDE(5, 1000, 0.0001)
inc_mat = m.incidence_matrix(a, score)

print((np.dot(inc_mat.T, inc_mat)).diagonal())