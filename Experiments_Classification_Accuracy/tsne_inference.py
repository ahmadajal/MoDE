import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat, savemat
import io
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def tsne_inference(test_data, train_data, X_2d_tsne, k):
    """
    This function generates t-SNE embeddings for out-of-sample exampels (test data)
    test_data: test data (note that test data should have gne under the same preprocessing steps as training data)
    train_data: training data
    X_2d_tsne: t-SNE embeddings of the training data
    k: number of nearest neighbors for t-SNE
    """
    #creating a k-nearest neighbor object on the train data
    print("computing {} nearest neighbors".format(k))
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train_data)
    # finding the k-neighbor indices for each of the test data
    distances, k_neighbour_indicies = neigh.kneighbors(test_data)
    # computing the t-sne embedding for each of the data points in test set
    # this is done by simply averaging the k-nearest neighbors' t-SNE embeddings
    # with weights being proportional to the inverse of the distances
    test_tsne = []
    for i in range(test_data.shape[0]):
        test_tsne.append(list(np.average(X_2d_tsne[k_neighbour_indicies[i]], axis=0, weights=distances[i]**-1)))
    return np.array(test_tsne)
