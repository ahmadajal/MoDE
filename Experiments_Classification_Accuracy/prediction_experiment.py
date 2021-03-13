"""
The script that generate the results for the "Performance in a classification task"
part in Section 4 (Experiments) of the paper
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat, savemat
import io
import json
import sklearn
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mode_inference import mode_inference
from tsne_inference import tsne_inference
import umap
import argparse
MATLAB_DATA_PATH = "../MATLAB_implementation/data/"
DATA_PATH = "../data/"
parser = argparse.ArgumentParser()
parser.add_argument('--mode_path', action="store")
parser.add_argument('--DM_path', action="store")
parser.add_argument('--ptsne_path', nargs="+", action="store")
parser.add_argument('--train_mat_path', action="store")
parser.add_argument('--test_mat_path', action="store")
parser.add_argument('--perplexity', type=float, action="store")

args = parser.parse_args()

train_data = loadmat(MATLAB_DATA_PATH+args.train_mat_path)["StockData"]
test_data = loadmat(MATLAB_DATA_PATH+args.test_mat_path)["StockData"]
if scipy.sparse.issparse(train_data):
    train_data = train_data.toarray()
    test_data = test_data.toarray()

train_data_score = loadmat(MATLAB_DATA_PATH+args.train_mat_path)["Score"].squeeze()
test_data_score = loadmat(MATLAB_DATA_PATH+args.test_mat_path)["Score"].squeeze()

train_data_labels = loadmat(MATLAB_DATA_PATH+args.train_mat_path)["labels"].squeeze()
test_data_labels = loadmat(MATLAB_DATA_PATH+args.test_mat_path)["labels"].squeeze()

# #load ptsne embeddings
# train_data_ptsne = loadmat(DATA_PATH+args.ptsne_path[0])["mapped_train_X"]
# test_data_ptsne = loadmat(DATA_PATH+args.ptsne_path[1])["mapped_test_X"]

# compting the number of nearest neighbors given the perplexity value
n_neighbors = min(len(train_data_labels) - 1, int(3. * args.perplexity + 1))
# find number of classes
nb_classes = len(set(train_data_labels))

# train_data_mode = loadmat(DATA_PATH+args.mode_path)["X_2d"]
# train_data_DM = loadmat(DATA_PATH+args.DM_path)["DM_average"]
# if scipy.sparse.issparse(train_data_mode):
#     train_data_mode = train_data_mode.toarray()
#     train_data_DM = train_data_DM.toarray()
# # t-SNE
# tsne = TSNE(n_components=2, perplexity=args.perplexity, n_iter=2000, random_state=None, verbose=1, metric="precomputed")
# train_data_tsne = tsne.fit_transform(train_data_DM)
# # ISOMAP
# isomap = Isomap(n_neighbors=n_neighbors, n_components=2, max_iter=2000, n_jobs=-1)
# train_data_isomap = isomap.fit_transform(train_data)
# test_data_isomap = isomap.transform(test_data)
# UMAP
reducer = umap.UMAP(n_neighbors=n_neighbors)
train_data_umap = reducer.fit_transform(train_data)
test_data_umap = reducer.transform(test_data)
#temp
plt.scatter(train_data_umap[:, 0], train_data_umap[:,1], c=train_data_labels,
                   cmap=plt.cm.get_cmap("YlOrRd", nb_classes))
plt.title("UMAP")
plt.colorbar()
plt.savefig("../results/" + args.mode_path.split("_")[0] + ".jpg", format="jpeg");
# plot
# fig, ax = plt.subplots(2,2, figsize=(9, 9))
# if nb_classes > 10:
#     pallete = "YlOrRd"
# else:
#     pallete = "Set1"
# s1 = ax[0,0].scatter(train_data_tsne[:, 0], train_data_tsne[:,1], c=train_data_labels,
#                    cmap=plt.cm.get_cmap(pallete, nb_classes))
# ax[0,0].set_title("t-SNE")
# plt.colorbar(mappable=s1, ax=ax[0,0])
# ##
# s2 = ax[0,1].scatter(train_data_mode[:,0], train_data_mode[:,1],
#                    c=train_data_labels, cmap=plt.cm.get_cmap(pallete, nb_classes))
# ax[0,1].set_title("MoDE")
# plt.colorbar(mappable=s2, ax=ax[0,1])
# ##
# s3 = ax[1,0].scatter(train_data_ptsne[:,0], train_data_ptsne[:,1],
#                    c=train_data_labels, cmap=plt.cm.get_cmap(pallete, nb_classes))
# ax[1,0].set_title("pt-SNE")
# plt.colorbar(mappable=s3, ax=ax[1,0])
# ##
# s4 = ax[1,1].scatter(train_data_isomap[:,0], train_data_isomap[:,1],
#                    c=train_data_labels, cmap=plt.cm.get_cmap(pallete, nb_classes))
# ax[1,1].set_title("ISOMAP")
# plt.colorbar(mappable=s4, ax=ax[1,1])
# plt.savefig("../figures/" + args.mode_path.split("_")[0] + ".jpg", format="jpeg");

# # generating test MoDE embeddings
# test_data_mode = mode_inference(test_data=test_data, train_data=train_data,
#                                        X_2d_mode=train_data_mode, k=n_neighbors)
# # generating test t-SNE embeddings
# test_data_tsne = tsne_inference(test_data=test_data, train_data=train_data,
#                                 X_2d_tsne=train_data_tsne, k=n_neighbors)
# prediction
results_test = {"full_data": {},
"MoDE": {},
"tSNE": {},
"ptsne": {},
"isomap": {},
"umap": {}}

results_train = {"full_data": {},
"MoDE": {},
"tSNE": {},
"ptsne": {},
"isomap": {},
"umap": {}}

LR = LogisticRegressionCV(Cs=np.logspace(-5, 5, 10), max_iter=4000, n_jobs=-1, verbose=0)
KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

LR.fit(train_data, train_data_labels)
results_test["full_data"]["LR"] = LR.score(test_data, test_data_labels)
print("LR accuracy for full data; ", results_test["full_data"]["LR"])
results_train["full_data"]["LR"] = LR.score(train_data, train_data_labels)
KNN.fit(train_data, train_data_labels)
results_test["full_data"]["KNN"] = KNN.score(test_data, test_data_labels)
print("KNN accuracy for full data; ", results_test["full_data"]["KNN"])
results_train["full_data"]["KNN"] = KNN.score(train_data, train_data_labels)


LR.fit(train_data_mode, train_data_labels)
results_test["MoDE"]["LR"] = LR.score(test_data_mode, test_data_labels)
print("LR accuracy for MoDE; ", results_test["MoDE"]["LR"])
results_train["MoDE"]["LR"] = LR.score(train_data_mode, train_data_labels)
KNN.fit(train_data_mode, train_data_labels)
results_test["MoDE"]["KNN"] = KNN.score(test_data_mode, test_data_labels)
print("KNN accuracy for MoDE; ", results_test["MoDE"]["KNN"])
results_train["MoDE"]["KNN"] = KNN.score(train_data_mode, train_data_labels)


LR.fit(train_data_tsne, train_data_labels)
results_test["tSNE"]["LR"] = LR.score(test_data_tsne, test_data_labels)
print("LR accuracy for t-SNE; ", results_test["tSNE"]["LR"])
results_train["tSNE"]["LR"] = LR.score(train_data_tsne, train_data_labels)
KNN.fit(train_data_tsne, train_data_labels)
results_test["tSNE"]["KNN"] = KNN.score(test_data_tsne, test_data_labels)
print("KNN accuracy for t-SNE; ", results_test["tSNE"]["KNN"])
results_train["tSNE"]["KNN"] = KNN.score(train_data_tsne, train_data_labels)

LR.fit(train_data_ptsne, train_data_labels)
results_test["ptsne"]["LR"] = LR.score(test_data_ptsne, test_data_labels)
print("LR accuracy for pt-SNE; ", results_test["ptsne"]["LR"])
results_train["ptsne"]["LR"] = LR.score(train_data_ptsne, train_data_labels)
KNN.fit(train_data_ptsne, train_data_labels)
results_test["ptsne"]["KNN"] = KNN.score(test_data_ptsne, test_data_labels)
print("KNN accuracy for pt-SNE; ", results_test["ptsne"]["KNN"])
results_train["ptsne"]["KNN"] = KNN.score(train_data_ptsne, train_data_labels)

LR.fit(train_data_isomap, train_data_labels)
results_test["isomap"]["LR"] = LR.score(test_data_isomap, test_data_labels)
print("LR accuracy for ISOMAP; ", results_test["isomap"]["LR"])
results_train["isomap"]["LR"] = LR.score(train_data_isomap, train_data_labels)
KNN.fit(train_data_isomap, train_data_labels)
results_test["isomap"]["KNN"] = KNN.score(test_data_isomap, test_data_labels)
print("KNN accuracy for ISOMAP; ", results_test["isomap"]["KNN"])
results_train["isomap"]["KNN"] = KNN.score(train_data_isomap, train_data_labels)

LR.fit(train_data_umap, train_data_labels)
results_test["umap"]["LR"] = LR.score(test_data_umap, test_data_labels)
print("LR accuracy for UMAP; ", results_test["umap"]["LR"])
results_train["umap"]["LR"] = LR.score(train_data_umap, train_data_labels)
KNN.fit(train_data_umap, train_data_labels)
results_test["umap"]["KNN"] = KNN.score(test_data_umap, test_data_labels)
print("KNN accuracy for UMAP; ", results_test["umap"]["KNN"])
results_train["umap"]["KNN"] = KNN.score(train_data_umap, train_data_labels)

# save the results
with open("../results/"+args.mode_path.split("_")[0]+ "_train" + ".json", "w") as f:
    json.dump(results_train, f)

with open("../results/"+args.mode_path.split("_")[0]+ "_test" + ".json", "w") as f:
    json.dump(results_test, f)
