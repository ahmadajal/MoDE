import numpy as np
from MoDE_embeddings.metrics import correlation_metric, distance_metric
from MoDE_embeddings.MoDE import MoDE
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE, Isomap
from umap import UMAP

# Load the swiss roll dataset
data, score = make_swiss_roll(n_samples=1000, random_state=1)

# Define the number of neighbors for each algorithm
num_neighbors = 20
# Dictionaries to keep the metric results
R_d = {}
R_c = {}

# Define MoDE embedding class
mode = MoDE(n_neighbor=num_neighbors, n_components=2)
x_2d_mode = mode.fit_transform(data, score)
# Compute metrics
R_d["MoDE"] = distance_metric(data, x_2d_mode, n_neighbor=num_neighbors)
R_c["MoDE"] = correlation_metric(data, x_2d_mode, n_neighbor=num_neighbors)

# t-SNE
tsne = TSNE(n_components=2, perplexity=num_neighbors // 3)
x_2d_tsne = tsne.fit_transform(data)
# Compute metrics
R_d["t-SNE"] = distance_metric(data, x_2d_tsne, n_neighbor=num_neighbors)
R_c["t-SNE"] = correlation_metric(data, x_2d_tsne, n_neighbor=num_neighbors)

# ISOMAP
isomap = Isomap(n_components=2, n_neighbors=num_neighbors)
x_2d_isomap = isomap.fit_transform(data)
# Compute metrics
R_d["Isomap"] = distance_metric(data, x_2d_isomap, n_neighbor=num_neighbors)
R_c["Isomap"] = correlation_metric(data, x_2d_isomap, n_neighbor=num_neighbors)

# UMAP
umap_model = UMAP(n_components=2, n_neighbors=num_neighbors)
x_2d_umap = umap_model.fit_transform(data)
# Compute metrics
R_d["UMAP"] = distance_metric(data, x_2d_umap, n_neighbor=num_neighbors)
R_c["UMAP"] = correlation_metric(data, x_2d_umap, n_neighbor=num_neighbors)

# Round the results
R_d = {k: np.round(v, 2) for k, v in R_d.items()}
R_c = {k: np.round(v, 2) for k, v in R_c.items()}


print(
    f"""
    MoDE: R_d = {R_d['MoDE']}, R_c = {R_c['MoDE']}
    t-SNE: R_d = {R_d['t-SNE']}, R_c = {R_c['t-SNE']}
    Isomap: R_d = {R_d['Isomap']}, R_c = {R_c['Isomap']}
    UMAP: R_d = {R_d['UMAP']}, R_c = {R_c['UMAP']}
    """
)
