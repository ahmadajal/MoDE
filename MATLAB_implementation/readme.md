### MATLAB implementation for MoDE:
This folder contains the MATLAB code used to generate the MoDE and the baseline embeddings (ISOMAP, MDS, and t-SNE). For the baseline embeddings, implementations from standard toolboxes were used. Some of the important scripts and folders in this folder:

- runme.m: By running this script you can generate 2D embeddigns for MoDE and the baseline methods. This script also generates the 2D visualization of these methods. By running this script you can generate a result similar to Figure 3 in the paper.

- "cv" folder: This folder contains the code for MoDE algorithm. The main function that generates the MoDE embeddings given the input data, is in the script "CV2.m". Also the script "MetricResultCalculation.m" contain the function that compute the preservation metrics R_d, R_c, and R_s for any 2D embedding.

- "data" folder: This folder contains the datasets we used in this paper in ".mat" format.

- "waterfill" folder: This folder contains the code for the data compresion algorithm we used. For more info on this algorithm see: https://arxiv.org/abs/1405.5873

