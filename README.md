# Multi-objective-2D-Embeddings (MoDE)
This repository contains the code and results for the paper "An Interpretable Data Embedding under Uncertain Distance Information". This paper presents a data embedding method called Multi-objective 2D Embedding (__MoDE__) that can successfully capture, with high fidelity, multiple facets of the data relationships: correlations, distances, and orders or importance rankings.


## Usage
__MoDE__ embeddings can be trained on exact or inexact distance matrices. In the case of inexact distance information, ranges of lower and upper bounds on the distances in the form of seperate lower and upper bound distance matrices should be given to the `fit_transform` function. The resulting embeddings are in 2D dimensions and the data points are placed in the embedding space such that samples with higher scores are placed in higher angles (in polar coordinates).
```
from MoDE import MoDE
mode = MoDE(n_neighbor=20, max_iter=100000, tol=0.0001, verbose=True)
x_2d = mode.fit_transform(data, score, dm_ub, dm_lb)
```
Once the MoDE embeddings are trained, you can measure the fidelity of the embedded dataset to the original dataset in terms of preserving distances, correlations and orders. To do so you can use the metric functions available in "metrics.py".
```
from metrics import distance_metric, correlation_metric, order_preservation
dm_avg = (dm_ub+dm_lb) / 2
R_d = distance_metric(data, x_2d, dm_avg, n_neighbor=20)
R_c = correlation_metric(data, x_2d, dm_avg, n_neighbor=20)
R_o = order_preservation(x_2d, dm_avg, n_neighbor=20, score=score.squeeze())
```
Below you can see the visulaization of MoDE embeddings for a dataset of stocks with 2252 samples and 1024 features. The market capitalization of each stock was used as score. You can also see the values of distance, correlation, and order metrics on top of the plot.

![Alt text](https://github.com/ahmadajal/Multi-objective-2D-Embeddings/blob/master/images/mode.png?raw=True)

## Waterfilling algorithm (for data compression)
With waterfilling algorithm you can find tight lower and upper bound on the pair-wise distances between data points that have been compressed using orthonormal transforms, e.g, fourier transform. Using the `WaterfillingCompression` class you can compress the data by keeping only a small portion of fourier transform coefficients. Then by calling the `compute_distance_bounds` method you are able to compute tight bounds on the lower and upper bound on distances.
```
from waterfilling_compression import WaterfillingCompression
comp = WaterfillingCompression(num_coeffs=4, coeffs_to_keep='optimal')
dm_ub, dm_lb = comp.compute_distance_bounds(data)
```

## Benchmarks to compare Python implementation with the MATLAB one
Below we will compare the Python implementation with the MATLAB one in terms of the metric values and the runtime of MoDE for different datasets.

### Distance, Correlation, and Order metrics
The table below shows the metrics accuracy results for Python and MATLAB implementation of MoDE.

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th colspan="3">Python</th>
    <th colspan="3">MATLAB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Metrics</td>
    <td>R_d</td>
    <td>R_o</td>
    <td>R_c</td>
    <td>R_d</td>
    <td>R_o</td>
    <td>R_c</td>
  </tr>
  <tr>
    <td>Small Stock</td>
    <td>0.708</td>
    <td>0.955</td>
    <td>0.864</td>
    <td>0.708</td>
    <td>0.960</td>
    <td>0.867</td>
  </tr>
  <tr>
    <td>Big Stock</td>
    <td>0.707</td>
    <td>0.952</td>
    <td>0.89</td>
    <td>0.707</td>
    <td>0.953</td>
    <td>0.894</td>
  </tr>
</tbody>
</table>

### Runtime comparison
The table below shows the runtime comparison (in seconds) for different datasets. The experiments were done on a 2.5 GHz 14-Core Intel Xenon with 256 GB of RAM.

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th># points</th>
    <th>Python</th>
    <th>MATLAB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Small Stock<br></td>
    <td>436</td>
    <td>8.14</td>
    <td>3.579</td>
  </tr>
  <tr>
    <td>Big Stock</td>
    <td>2256</td>
    <td>133.5</td>
    <td>69.768</td>
  </tr>
  <tr>
    <td>Breast Cancer</td>
    <td>569</td>
    <td>11.1</td>
    <td>4.56</td>
  </tr>
  <tr>
    <td>cifar-10 (subset)</td>
    <td>8000</td>
    <td>838.81</td>
    <td>409.52</td>
  </tr>
  <tr>
    <td>EEG</td>
    <td>11853</td>
    <td>2594.15</td>
    <td>583.323</td>
  </tr>
  <tr>
    <td>heart beat</td>
    <td>14545</td>
    <td>1529.77</td>
    <td>837.88</td>
  </tr>
  <tr>
    <td>madelon</td>
    <td>2080</td>
    <td>147.25</td>
    <td>65.75</td>
  </tr>
  <tr>
    <td>MNIST (subset)</td>
    <td>2000</td>
    <td>96.02</td>
    <td>23.89</td>
  </tr>
</tbody>
</table>
