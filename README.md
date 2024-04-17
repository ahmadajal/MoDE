# MoDE: Multi-objective-Data-Embedding

This repository contains the code and results for the paper **"An Interpretable Data Embedding under Uncertain Distance Information"**, published at the International Conference on Data Mining (ICDM) in 2020. 

Below you may see a nice visualization of the iterations of MoDE that show the convergence of the algorithm for the well-known [S-curve](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html) dataset.

<p align="middle">
  <img src="https://github.com/ahmadajal/MoDE/blob/master/Python_implementation/S_shape_data_org.jpg" width="300" />
  <img src="https://github.com/ahmadajal/MoDE/blob/master/Python_implementation/MoDE_iterations.gif" width="500" /> 
</p>

To get a glimpse of the advantages of using MoDE in data visualization, you may watch the conference presentation:

[![](http://img.youtube.com/vi/WC6ESPrQLXo/0.jpg)](http://www.youtube.com/watch?v=WC6ESPrQLXo "Video Presentation")


# Details

Multi-objective Data Embedding (__MoDE__) is a 2D data embedding that captures, with high fidelity, multiple facets of the data relationships: 

- correlations, 
- distances, and, 
- orders or importance rankings. 

An unique characteristic of MoDE is that it does not require exact distances between the objects, like most visualization techniques do. We can give ranges of lower and upper bound distances between objects, which means that MoDE can **effectively visualize compressed or uncertain data**!

Moreover, this embedding method enhances **interpretability** because:

1) It incorporates the ranks or scores of the data samples (if such ranks exist in the dataset) in the resulting embeddings and by placing points with higher scores in higher angles in 2D, provides an interpretable data embedding. 
2) The embedding typically results in a "half-moon" visualization of the data. Therefore, the user sees typically a similar visualization of the data so understanding and interpretation is easier. For many other techniques, not only each dataset provides a different visualization outcome, but also different runs of the visualization method may give different visualization results.

In recent work we have also extended __MoDE__ to work not only on 2D, but to project on any dimensionality. 

# Useful Links
- The [conference paper](https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/MoDE_ICDM_2020.pdf) at ICDM 2020, and an extended version has been accepted in the [ACM Transactions on Knowledge Discovery from Data (TKDD)](https://dl.acm.org/doi/abs/10.1145/3537901) [2] which includes the extension of MoDE to n-dimensional embeddings. 
- Datasets used for the experiments in the paper are [here](https://www.dropbox.com/sh/r5ovlq82ihcpc1j/AAALX__nRzVOShJMfhj35ZJBa?dl=0).

This repository contains both the Python and MATLAB implementations of MoDE. __Note that you can replicate the experimental results in the ICDM paper, using the MATLAB implementation of MoDE.__

Below you can see the visulaization of MoDE embeddings for a dataset of stocks with 2252 samples and 1024 features. The market capitalization of each stock was used as the "rank" of each stock: higher rank will place the object at a higher angular position. You can also see the values of distance, correlation, and order preservation metrics on top of the plot.

- **$R_d$** shows how well pairwise distances are captured (1 is best).
- **$R_o$** shows how well "orders", or ranks, are captured (1 is best).
- **$R_c$** shows how well correlations are captured (1 is best).

<img src="https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/images/mode.png?raw=True" alt="mode_image" width="500">


If you find this code useful or use it in a publication or research, please cite [1,2]. We would also love to hear how you have used this code.

## References
[1] N. Freris, M. Vlachos, A. Ajalloeian: "An Interpretable Data Embedding under Uncertain Distance Information", Proc. of IEEE ICDM 2020

[2] N. Freris, A. Ajalloeian, M. Vlachos: "Interpretable Embedding and Visualization of Compressed Data", ACM Transactions on Knowledge Discovery from Data (TKDD)
## Description of the folders
- _"Experiment_Classification_Accuracy":_ This folder contains the code for the experiments which were conducted to compare MoDE with baseline embeddings in terms of the accuracy of a classification task (When the model was trained on the embeddings).

- _"MATLAB_implementation":_ This folder contains the MATLAB implementation of MoDE.

- _"Python_implemetation":_ This folder contains the Python implementation of MoDE.

- _"benchmark":_ This folder contains the code for benchmarking the MATLAB and Python implementations of MoDE in terms of results and execution time.
