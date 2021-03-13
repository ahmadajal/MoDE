#!/bin/bash
python run_MoDE_ndim.py --dataset breast_cancer_train --dims 2 6 10 14 18 22 26 30
python run_MoDE_ndim.py --dataset wine_train --dims 2 4 6 8 10 11
python run_MoDE_ndim.py --dataset wafer_train --dims 2 4 8 16 32 64 96 128
python run_MoDE_ndim.py --dataset small_stock --dims 2 4 8 16 32 64 96 128
python run_MoDE_ndim.py --dataset arrow_train --dims 2 4 8 16 32 64 96 128 192 256 384 512 1024
python run_MoDE_ndim.py --dataset phishing_train --dims 2 4 8 16 24 34 44 54 68