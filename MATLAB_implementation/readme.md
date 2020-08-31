# MoDE
This folder contains the code and results for the paper "An Interpretable Data Embedding under Uncertain Distance Information". This paper presents a data embedding method called Multi-objective 2D Embedding (__MoDE__) that can successfully capture, with high fidelity, multiple facets of the data relationships: correlations, distances, and orders or importance rankings. This folder contains 4 sub-folders.

### Experiments_Embedding_Quality:
This folder contains the MATLAB code used to generate the MoDE and the baseline embeddings (ISOMAP, MDS, and t-SNE). For the baseline embeddings, implementations from standard toolboxes were used. Some of the important scripts and folders in this folder:

- runme.m: By running this script you can generate 2D embeddigns for MoDE and the baseline methods. This script also generates the 2D visualization of these methods. By running this script you can generate a result similar to Figure 3 in the paper.

- "cv" folder: This folder contains the code for MoDE algorithm. The main function that generates the MoDE embeddings given the input data, is in the script "CV2.m". Also the script "MetricResultCalculation.m" contain the function that compute the preservation metrics R_d, R_c, and R_s for any 2D embedding.

- "data" folder: This folder contains the datasets we used in this paper in ".mat" format.

- "waterfill" folder: This folder contains the code for the data compresion algorithm we used. For more info on this algorithm see: https://arxiv.org/abs/1405.5873

### Experiments_Classification_Accuracy:
This folder contains the Python code used for the classification accuracy task (Section 4.B). For classification algorithms we used standard implementations from scikit-learn package.Some of the important scripts in this folder:

- prediction_experiment.py: By running this script, you can generate the classification accuracy results, provided that proper input arguments are given to the script.

- mode_inference.py and tsne_inference.py: Functions that generate MoDE and t-SNE embeddings for out of sample examples (test data). For more details on how this is done see Section 4.B of the paper (equation (8)).

### Experiments_Scalability:
This folder only contains the result of the scalability experiment. The notebook "scalability plot.ipynb" contains the plot that highlights the linear scalability of MoDE (Figure 6).
