import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import pickle
import sys
from MoDE import MoDE
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from metrics import distance_metric, correlation_metric, order_preservation
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', action="store", default="small_stock.mat")
parser.add_argument('--normalize', action="store", default="No")
args = parser.parse_args()

data = loadmat(args.data_path)["StockData"]
score = loadmat(args.data_path)["Score"]

# save the ouput log file
LOG_PATH = "logs/"
sys.stdout = open(LOG_PATH + "log_" + args.data_path.split(".")[0] + ".txt", 'w')
# empty the buffer
sys.stdout.flush()

if args.normalize == "standard":
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
elif args.normalize == "min-max":
    m = np.mean(data, axis=1)
    data = data - m.reshape((-1, 1))

    s = np.max(data, axis=1) - np.min(data, axis=1)
    data = data / s.reshape((-1, 1))
elif args.normalize == "No":
    pass
else:
    raise Exception("error: wrong normalization argument")

print("data shape: ", data.shape)
# empty the buffer
sys.stdout.flush()

dm = pairwise_distances(data, n_jobs=-1)
# temporary: for now limit the decimals
dm = np.round(dm, decimals=5)
mode = MoDE(n_neighbor=20, max_iter=100000, tol=0.0001, verbose=True)

start = time.time()

x_2d_mode = mode.fit_transform(data, score.squeeze(), dm, dm)

print("time: ", time.time() - start)

