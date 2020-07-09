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

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Dataset</th>
    <th class="tg-c3ow" colspan="3">Python</th>
    <th class="tg-baqh" colspan="3">MATLAB</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:bold">__Metrics__</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">R_d</span></td>
    <td class="tg-c3ow"><span style="font-weight:bold">R_o</span></td>
    <td class="tg-0lax"><span style="font-weight:bold">R_c</span></td>
    <td class="tg-0lax"><span style="font-weight:bold">R_d</span></td>
    <td class="tg-0lax"><span style="font-weight:bold">R_o</span></td>
    <td class="tg-0lax"><span style="font-weight:bold">R_c</span></td>
  </tr>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">Small Stock</span></td>
    <td class="tg-c3ow">0.708</td>
    <td class="tg-c3ow">0.955</td>
    <td class="tg-0lax">0.864</td>
    <td class="tg-0lax">0.708</td>
    <td class="tg-0lax">0.960</td>
    <td class="tg-0lax">0.867</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Big Stock</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
    <td class="tg-0lax"></td>
  </tr>
</tbody>
</table>


