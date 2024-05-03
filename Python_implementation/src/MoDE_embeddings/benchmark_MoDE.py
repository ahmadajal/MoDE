import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from collections import OrderedDict
from functools import partial
from time import time
from MoDE_embeddings import MoDE
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.metrics import pairwise_distances
from MoDE_embeddings.metrics import distance_metric, correlation_metric, order_preservation
from sklearn import datasets, manifold
####
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--n_points', type=int, default=10000, help='number of data points')
args = argparser.parse_args()
####

n_points = 100000 # take subsets of 100K points
n_neighbors = 10
n_components = 2

data, color = datasets.make_swiss_roll(n_points, random_state=93)
np.random.seed(93)
inds = np.arange(0, n_points)
np.random.shuffle(inds)
# take subsets
data = data[inds[:args.n_points]]
color = color[inds[:args.n_points]]

# data = loadmat("/Users/ahmad/MoDE/data/madelon_train.mat")["StockData"]

# if scipy.sparse.issparse(data):
#     data = data.toarray()
    
# color = loadmat("/Users/ahmad/MoDE/data/madelon_train.mat")["Score"].squeeze()

dm = pairwise_distances(data, n_jobs=-1)
# temporary: for now limit the decimals
dm = np.round(dm, decimals=5)

# Set-up manifold methods
LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors, n_components, eigen_solver='auto')

methods = OrderedDict()
# methods['LLE'] = LLE(method='standard')
# methods['LTSA'] = LLE(method='ltsa')
# methods['Hessian LLE'] = LLE(method='hessian')
# methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
# #methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
# methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
#                                            n_neighbors=n_neighbors)
# methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
#                                  random_state=0)
methods['MoDE'] = MoDE(n_neighbor=n_neighbors, max_iter=100000, tol=0.001, verbose=True)


# Plot results
for i, (label, method) in enumerate(methods.items()):
    t0 = time()

    if label == 'MoDE':
      Y = method.fit_transform(data, color)
    else:
      Y = method.fit_transform(data)
    R_d = distance_metric(data,Y, n_neighbor=n_neighbors)
    R_c = correlation_metric(data, Y, n_neighbor=n_neighbors)
    t1 = time()
    print("%s: %.4g sec" % (label, t1 - t0))
    print("%s: R_d = %.5g, R_c = %.5g " % (label, R_d, R_c))
