# Multi-objective-2D-Embeddings
The source code for MoDE:

Things that needs to be done:
- The current results looks strange. See the `test_MoDE.ipynb` notebook. The visualization for "small_stock" dataset looks strange
- implementation of the distance, correlation, and order metrics.
- Implementation of the compression algorithm.

## Usage
```
from MoDE import MoDE
mode = MoDE(n_neighbor=20, max_iter=100000, tol=0.0001, verbose=True)
x_2d = mode.fit_transform(data, score, dm, dm)
```
