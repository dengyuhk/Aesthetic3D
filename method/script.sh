#!/bin/bash

#redo training
python train_split.py --log_path Logs --batchSize 128 --save_rate 5 --architecture ModelImg2 --data_path ./data/pre_rendered_3.pt --pairs_path ./data/pairs_weighted_filtered.pt

#infer
python eval_vis.py --checkpoint ./Logs/paras_50.pth --mesh_path ./test.obj