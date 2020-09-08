# Multi-objective-2D-Embeddings (MoDE)
This repository contains the code and results for the paper **"An Interpretable Data Embedding under Uncertain Distance Information"**, published at the International Conference on Data Mining (ICDM) in 2020. 

The work presents a data embedding method called Multi-objective 2D Embedding (__MoDE__) that captures, with high fidelity, multiple facets of the data relationships: correlations, distances, and orders or importance rankings. Moreover, this embedding method enhances **interpretability** because:

1) It incorporates the ranks or scores of the data samples (if such ranks exist in the dataset) in the resulting embeddings and by placing points with higher scores in higher angles in 2D, provides an interpretable data embedding. 
2) The embedding typically results in a "half-moon" visualization of the data. Therefore, the user sees typically a similar visualization of the data so understanding and interpretation is easier. For many other techniques, not only each dataset provides a different visualization outcome, but also different runs of the visualization method may give different visualization results.

Useful Links:
- An [extended version](https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/MoDE_ICDM.pdf) of the ICDM publication (with additional experiments, proofs, etc). 
- Datasets used for the experiments in the paper are [here](https://www.dropbox.com/sh/r5ovlq82ihcpc1j/AAALX__nRzVOShJMfhj35ZJBa?dl=0).

This repository contains both the Python and MATLAB implementations of MoDE. __Note that you can replicate the experimental results in the ICDM paper, using the MATLAB implementation of MoDE.__

Below you can see the visulaization of MoDE embeddings for a dataset of stocks with 2252 samples and 1024 features. The market capitalization of each stock was used as score. You can also see the values of distance, correlation, and order preservation metrics on top of the plot.

![Alt text](https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/images/mode.png?raw=True)

If you find this code useful or use it in a publication or research, please use citation [1]. We would also love to hear how you have used this code.

## References
[1] N. Freris, M. Vlachos, A. Ajalloeian: "An Interpretable Data Embedding under Uncertain Distance Information", Proc. of ICDM 2020

## Description of the folders
- _"Experiment_Classification_Accuracy":_ This folder contains the code for the experiments which were conducted to compare MoDE with baseline embeddings in terms of the accuracy of a classification task (When the model was trained on the embeddings).

- _"MATLAB_implementation":_ This folder contains the MATLAB implementation of MoDE.

- _"Python_implemetation":_ This folder contains the Python implementation of MoDE.

- _"benchmark":_ This folder contains the code for benchmarking the MATLAB and Python implementations of MoDE in terms of results and execution time.
