# Python implementation for MoDE (Multi-objective Data Embedding)
## Description of scripts in `MoDE_embeddings/`
- __`MoDE.py`__: This file contains the main class that implements MoDE.
- __`metrics.py`__: This file contains the functions to compute the three metrics introduced in the paper, i.e, distance, correlation, and order preservation metrics.
- __`waterfilling_compression.py`__: This file contains the implementation of waterfilling algorithm.
- __`fastgd/`__: This directory contains the fast implementation of the Gradient Decsent algorithm in Cython.
## Usage
__MoDE__ embeddings can be trained on exact or inexact distance matrices. In the case of inexact distance information, ranges of lower and upper bounds on the distances in the form of seperate lower and upper bound distance matrices should be given to the `fit_transform` function. The resulting embeddings are in 2D dimensions and the data points are placed in the embedding space such that samples with higher scores are placed in higher angles (in polar coordinates).
```
from MoDE_embeddings.MoDE import MoDE
mode = MoDE(n_neighbor=20, max_iter=100000, tol=0.0001, verbose=True)
x_2d = mode.fit_transform(data, score)
```
Once the MoDE embeddings are trained, you can measure the fidelity of the embedded dataset to the original dataset in terms of preserving distances, correlations and orders. To do so, you can use the metric functions available in "metrics.py".
```
from MoDE_embeddings.metrics import distance_metric, correlation_metric, order_preservation
R_d = distance_metric(data, x_2d, n_neighbor=20)
R_c = correlation_metric(data, x_2d, n_neighbor=20)
R_o = order_preservation(x_2d, mode.P.squeeze(), n_neighbor=20, score=score.squeeze())
```

## Waterfilling algorithm (for data compression)
With the waterfilling algorithm you can find tight lower and upper bounds on the pair-wise distances between data points that have been compressed using orthonormal
transforms, e.g, fourier transform. Using the `WaterfillingCompression` class you can compress the data by keeping only a small portion of fourier transform
coefficients. Then by calling the `compute_distance_bounds` method you are able to compute tight lower and upper bounds on pair-wise distances. For more information
on the waterfilling algorithm check out the paper: https://arxiv.org/pdf/1405.5873.pdf
```
from MoDE_embeddings.waterfilling_compression import WaterfillingCompression
comp = WaterfillingCompression(num_coeffs=4, coeffs_to_keep='optimal')
dm_ub, dm_lb = comp.compute_distance_bounds(data)
```
