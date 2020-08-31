# FIDE (Fast and Interpretable 2D Embedding)

This code implements **FIDE** algorithm, and compares with other leading embedding schemes. 

**FIDE (Fast and Interpretable 2D Embedding)** is a data embedding method that seeks to preserve distance, correlation and rank order from the original space. The details can be found in our paper "FIDE: Fast and Interpretable 2D Embedding with correlation, distance and rank considerations".

On Code Ocean, pressing run will execute demo.m, which runs FIDE on 2016 University Ranking dataset and produce its 2d embedding; Performance metrics are also calculated w.r.t. distance, correlation and rank order.

### Functions in this capsule
* FIDE: This function performs FIDE on X to generate a 2D embedding using Gd.

* FIDE_Gc: This function performs FIDE on X to generate a 2D embedding using Gc.

* adjmat: This function creates the adjacency matrix of a graph from dissimality matrix, used in FIDE.

* incmat: This function creates the incidence matrix of a graph from adjacency matrix, used in FIDE

* DLS: This function implements the Distributed Linear Solver (DLS) algorithm, used in FIDE.

* **Experiment Folder**: This folder contains the code for experiments on various datasets from different application domains. The code executes FIDE and other methods to different datasets and returns results for the 3 performance metrics, as explicated in the paper.

Comparison of algorithms used in experiments are adapted from: 
MDS & tSNE:
Matlab Toolbox for Dimensionality Reduction,
Author: Laurens van der Maaten,
Affiliation: Delft University of Technology,
Contact: lvdmaaten@gmail.com,
Release date: March 21, 2013
Version: 0.8.1b

Isomap: 
Author: utkarsh trivedi (2017), MATLAB Central File Exchange,
Source: https://www.mathworks.com/matlabcentral/fileexchange/62449-isomap-d-n_fcn-n_size-options, Version: 1.0

LLE:
Author: Sam T. Roweis
Source: https://cs.nyu.edu/~roweis/lle/code.html

* FeatureScale: This function preprocesses data using min-max normalization.

* MetricResultCalculation: This function measures embedding performance by outputting 3 metric results R_s, R_c, R_d computed by comparing nearest neighbors before-and-after embedding data.

* Preprocess_data: This script provides information on how each dataset is preprocessed.

* All_Experiments: This script will perform every experiment from the separate experiment files. 

* plot: This script generate several related plot for FIDE.

### How to use FIDE
There is a number of parameters that can be set for FIDE; the main ones are as follows:

* k_FIDE: This determines the number of neighboring points used in k-NNG, for the neighborhood topology in the original space. The default value for this parameter is set as 5% of number of instances for small size datasets, and in a range 5 to 50 for larger ones.

* Preprocess: This decides whether data need to be feature-scaled using min-max normalization.

* MaxIter & Precision: Maximum number of iterations (as explicated in the paper) & Precision for the stopping criterion of DLS; in general, precision is set to 10^(-3) or 10^(-4);

### Data

* Times World University Rankings (2011-2016): This data includes university ranking information and corresponding performance indicators between 2011 and 2016. This data is obtained from [Times Higher Education](https://www.timeshighereducation.com/world-university-rankings). The methodology and description of paramters can be viewed at [here](https://www.timeshighereducation.com/world-university-rankings/methodology-world-university-rankings-2016-2017).

* HappinessAlcoholConsumption: This data includes happiness, IDH and GDP score versus the alcohol per capita consumption (by country and kind of drink). This data is obtained from [data.world](https://www.kaggle.com/marcospessotto/happiness-and-alcohol-consumption).

* winequality-red/winequality-white: This data describes red and white variants of the Portuguese "Vinho Verde" wine. This data is obtained from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), attribute information of this dataset can be found at [Cortez et al., 2009].

* stock_prices: stock prices of 100 days approximately between Aug 2018 and January 2019 for 450 stocks in NASDAQ extracted using [Alpha Vantage API](https://github.com/RomelTorres/alpha_vantage). The folder "stock_prices" contains one filename per stock, the filename is ticker name of the stock ticker + underscore + market capitization of the company in billions.
