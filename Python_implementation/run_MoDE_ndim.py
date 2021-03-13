import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import pickle
from MoDE import MoDE
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.metrics import pairwise_distances
import umap
from metrics import distance_metric, correlation_metric, order_preservation
import pickle
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, help='dataset name')
argparser.add_argument('--dims', nargs='+', type=int,
                       help='the list of the dimensions for the experiment')
argparser.add_argument('--k', type=int, default=20, help='number of nearest_neighbors')
argparser.add_argument('--epsilon', type=float, default=1e-4, help='tolerance for MoDE')
args = argparser.parse_args()

output_dir = "MoDE_ndim_results_tune/eps_{}_k_{}/".format(args.epsilon, args.k)
if ~os.path.isdir(output_dir):
    os.mkdir(output_dir)
data = loadmat("../MATLAB_implementation/data/"+args.dataset+".mat")["StockData"]
score = loadmat("../MATLAB_implementation/data/"+args.dataset+".mat")["Score"]

# #normalize: only for small stock data
# m = np.mean(data, axis=1)
# data = data - m.reshape((-1,1))

# s = np.max(data, axis=1) - np.min(data, axis=1)
# data = data / s.reshape((-1,1))

print(data.shape)
N, dim = data.shape

n_neighbor=args.k

dm = pairwise_distances(data, n_jobs=-1)
dm = np.round(dm, decimals=5)
mode = MoDE(n_neighbor=n_neighbor, max_iter=100000, tol=args.epsilon, n_components=dim, verbose=True)
start = time.time()

x_p_mode = mode.fit_transform(data, score.squeeze(), dm, dm)

print("time: ", time.time() - start)
np.save(output_dir+args.dataset+"_Phi.npy", mode.P)

x_range=args.dims
x_p_tsne = {}
for i in x_range:
    tsne = TSNE(n_components=i, perplexity=n_neighbor/3, n_iter=3000, verbose=1, method='exact')
    x_p_tsne[i] = tsne.fit_transform(data)

x_p_isomap = {}
for i in x_range:
    isomap = Isomap(n_neighbors=n_neighbor, n_components=i)
    x_p_isomap[i] = isomap.fit_transform(data)

x_p_mds = {}
for i in x_range:
    mds = MDS(n_components=i)
    x_p_mds[i] = mds.fit_transform(data)

x_p_umap = {}
for i in x_range:
    reducer = umap.UMAP(n_neighbors=n_neighbor, n_components=i)
    x_p_umap[i] = reducer.fit_transform(data)

data_norms = np.linalg.norm(data, axis=1)
Rd_s = {"mode":[], "tsne":[], "isomap":[], "mds":[], "umap":[]}
Rc_s = {"mode":[], "tsne":[], "isomap":[], "mds":[], "umap":[]}
for p in x_range:
    x_p = mode.to_hyper_spherical(data_norms, mode.P[:, :p-1])
    Rd_s["mode"].append(distance_metric(data, x_p, dm, n_neighbor=n_neighbor))
    Rc_s["mode"].append(correlation_metric(data, x_p, dm, n_neighbor=n_neighbor))

    Rd_s["tsne"].append(distance_metric(data, x_p_tsne[p], dm, n_neighbor=n_neighbor))
    Rc_s["tsne"].append(correlation_metric(data, x_p_tsne[p], dm, n_neighbor=n_neighbor))

    Rd_s["isomap"].append(distance_metric(data, x_p_isomap[p], dm, n_neighbor=n_neighbor))
    Rc_s["isomap"].append(correlation_metric(data, x_p_isomap[p], dm, n_neighbor=n_neighbor))

    Rd_s["mds"].append(distance_metric(data, x_p_mds[p], dm, n_neighbor=n_neighbor))
    Rc_s["mds"].append(correlation_metric(data, x_p_mds[p], dm, n_neighbor=n_neighbor))

    Rd_s["umap"].append(distance_metric(data, x_p_umap[p], dm, n_neighbor=n_neighbor))
    Rc_s["umap"].append(correlation_metric(data, x_p_umap[p], dm, n_neighbor=n_neighbor))

fig, ax = plt.subplots(1,2 , figsize=(14, 6))

for method in Rd_s.keys():
    ax[0].plot(x_range, Rd_s[method], label=method)
    ax[1].plot(x_range, Rc_s[method], label=method)
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("dimension of the embedding")
ax[0].set_ylabel(r"$R_d$", fontsize=12)
ax[1].set_xlabel("dimension of the embedding")
ax[1].set_ylabel(r"$R_c$", fontsize=12)
plt.savefig(output_dir+args.dataset+"_metric_comparison.jpg", format="jpg", dpi=300)

with open(output_dir+args.dataset+"_metrics_d.pkl", "wb") as f:
    pickle.dump(Rd_s, f)
with open(output_dir+args.dataset+"_metrics_c.pkl", "wb") as f:
    pickle.dump(Rc_s, f)
