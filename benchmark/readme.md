# Benchmarking the Python and MATLAB implementation
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
    <td>0.077</td>
    <td>3.579</td>
  </tr>
  <tr>
    <td>Big Stock</td>
    <td>2256</td>
    <td>0.63</td>
    <td>69.768</td>
  </tr>
  <tr>
    <td>Breast Cancer</td>
    <td>569</td>
    <td>0.078</td>
    <td>4.56</td>
  </tr>
  <tr>
    <td>cifar-10 (subset)</td>
    <td>8000</td>
    <td>9.42</td>
    <td>409.52</td>
  </tr>
  <tr>
    <td>EEG</td>
    <td>11853</td>
    <td>11.8</td>
    <td>583.323</td>
  </tr>
  <tr>
    <td>heart beat</td>
    <td>14545</td>
    <td>19.14</td>
    <td>837.88</td>
  </tr>
  <tr>
    <td>madelon</td>
    <td>2080</td>
    <td>0.56</td>
    <td>65.75</td>
  </tr>
</tbody>
</table>

