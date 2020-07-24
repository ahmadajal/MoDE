#!/usr/bin/env bash
python benchmark_MoDE.py --data_path small_stock --normalize min-max
python benchmark_MoDE.py --data_path big_stock --normalize min-max