#!/usr/bin/env bash
python benchmark_MoDE.py --data_path small_stock --normalize min-max
python benchmark_MoDE.py --data_path big_stock --normalize min-max
python benchmark_MoDE.py --data_path ../MoDE/Experiments_MATLAB/data/breast_cancer.mat
python benchmark_MoDE.py --data_path ../MoDE/Experiments_MATLAB/data/cifar_train.mat
python benchmark_MoDE.py --data_path ../MoDE/Experiments_MATLAB/data/eeg.mat
python benchmark_MoDE.py --data_path ../MoDE/Experiments_MATLAB/data/hb.mat
python benchmark_MoDE.py --data_path ../MoDE/Experiments_MATLAB/data/madelon_train.mat
python benchmark_MoDE.py --data_path ../MoDE/Experiments_MATLAB/data/mnist.mat


