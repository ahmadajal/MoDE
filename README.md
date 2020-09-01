# Multi-objective-2D-Embeddings (MoDE)
This repository contains the code and results for the paper "An Interpretable Data Embedding under Uncertain Distance Information". This paper presents a data embedding method called Multi-objective 2D Embedding (__MoDE__) that can successfully capture, with high fidelity, multiple facets of the data relationships: correlations, distances, and orders or importance rankings. Moreover, this embedding method incorporates the ranks or scores of the data samples (if such ranks exist in the dataset) in the resulting embeddings and by placing points with higher scores in higher angles in 2D, provides an interpretable data embedding. For more information on MoDE check out the full paper: https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/MoDE_ICDM.pdf

You can also access the datasets we used for the experiments in the paper, [here](https://www.dropbox.com/sh/r5ovlq82ihcpc1j/AAALX__nRzVOShJMfhj35ZJBa?dl=0).

This repository contains both the Python and MATLAB implementations of MoDE. __Note that for generating the results in the paper, we used the MATLAB implementation of MoDE.__

Below you can see the visulaization of MoDE embeddings for a dataset of stocks with 2252 samples and 1024 features. The market capitalization of each stock was used as score. You can also see the values of distance, correlation, and order preservation metrics on top of the plot.

![Alt text](https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/images/mode.png?raw=True)

If you find this code useful or use it in a publication or research, please cite it as follows:

N. Freris, M. Vlachos, A. Ajalloeian: "An Interpretable Data Embeddingunder Uncertain Distance Information", Proc. of ICDM 2020
