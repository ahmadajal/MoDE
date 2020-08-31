import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat, savemat
import io
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def mode_inference(test_data, train_data, X_2d_mode, k):
    """
    This function generates MoDE embeddings for out-of-sample exampels (test data)
    test_data: test data (note that test data should have gne under the same preprocessing steps as training data)
    train_data: training data
    X_2d_mode: MoDE embeddings of the training data
    k: number of nearest neighbors for MoDE
    """
    #creating a k-nearest neighbour object on the train data
    print("computing {} nearest neighbors".format(k))
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train_data)
    # finding the k-neighbour indices for each of the test data
    distances, k_neighbour_indicies = neigh.kneighbors(test_data)
    # computing the MoDE embedding for each of the data points in test set
    # this is done by simply averaging the k-nearest neighbors' MoDE embeddings
    # with weights being proportional to the inverse of the distances
    test_mode = []
    for i in range(test_data.shape[0]):
        # averaging angels (deprecated)
        # angels = [cart2pol(x,y)[1] for x,y in X_2d_mode[k_neighbour_indicies[i]]]
        # theta = np.average(angels, weights = distances[i]**-1)
        # r = np.linalg.norm(test_data[i])
        # test_mode.append(list(pol2cart(r, theta)))
        # averaging points
        test_mode.append(list(np.average(X_2d_mode[k_neighbour_indicies[i]], axis=0, weights=distances[i]**-1)))
    return np.array(test_mode)

def mode_inference_deprecateed(test_data, train_data, X_2d_mode, k, test_score, train_score):
    """
    this function is deprecated!
    """
    #creating a k-nearest neighbour object on the train data
    print("computing {} nearest neighbors".format(k))
    neigh = NearestNeighbors(n_neighbors=k, p=2)
    neigh.fit(train_data)
    # finding the k-neighbour indices for each of the test data
    _, k_neighbour_indicies = neigh.kneighbors(test_data)
    # computing the angle for each of the data points in test set
    test_mode = []
    for i in range(test_data.shape[0]):
        phi_i = 0
        for j in k_neighbour_indicies[i]:
            _, phi_j = cart2pol(X_2d_mode[j, 0], X_2d_mode[j, 1])
            corr = np.inner(test_data[i], train_data[j])
            corr = corr / (np.linalg.norm(test_data[i]) * np.linalg.norm(train_data[j]))
            if train_score[j] <= test_score[i]:
                theta_diff = np.arccos(corr)
                phi_i = phi_i + phi_j + theta_diff
            else:
                theta_diff = np.arccos(corr)
                phi_i = phi_i + phi_j - theta_diff
        # angles.append(phi_i/k)
        x, y = pol2cart(np.linalg.norm(test_data[i]), phi_i/k)
        test_mode.append([x, y])
    return np.array(test_mode)
