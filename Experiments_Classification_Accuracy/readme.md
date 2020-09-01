### Classification accuracy experiments
This folder contains the Python code used for the classification accuracy task (Section 4.B). For classification algorithms we used standard implementations 
from [scikit-learn](https://scikit-learn.org/stable/) package. Some of the important scripts in this folder:

- "prediction_experiment.py": By running this script, you can generate the classification accuracy results, provided that proper input arguments are given to 
the script.

- "mode_inference.py" and "tsne_inference.py": Functions that generate MoDE and t-SNE embeddings for out of sample examples (test data). For more details on how
this is done see Section 4.B of the paper (equation (8)).
