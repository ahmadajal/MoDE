# Multi-objective-2D-Embeddings
The source code for MoDE. For now the code seems to be working fine, e.g, take a look at the "test_MoDE.ipynb" notebook to observe the visualization for "small_stock" dataset.

Things that needs to be done:
- implementation of the distance, correlation, and order metrics.
- Implementation of the compression algorithm.

## Usage
```
from MoDE import MoDE
mode = MoDE(n_neighbor=20, max_iter=100000, tol=0.0001, verbose=True)
x_2d = mode.fit_transform(data, score, dm, dm)
```

## Benchmarks to compare Python implementation with the MATLAB one
Below we will compare the Python implementation with the MATLAB one in terms of the metric values and the runtime of MoDE for different datasets.

### Distance, Correlation, and Order metrics

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
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>

