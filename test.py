import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import identity, find, csr_matrix
from scipy.io import loadmat
from MoDE import MoDE
import scipy



data = loadmat("data/small_stock.mat")["StockData"]
score = loadmat("data/small_stock.mat")["Score"]

#normalize
m = np.mean(data, axis=1)
data = data - m.reshape((-1,1))

s = np.max(data, axis=1) - np.min(data, axis=1)
data = data / s.reshape((-1,1))

print(data.shape)
from waterfilling_compression import WaterfillingCompression
comp = WaterfillingCompression()
dm_ub, dm_lb = comp.compute_distance_bounds(data)
print(dm_ub[0])

