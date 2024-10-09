#!/bin/bash

set -x

python run_eval.py --base_architecture dinov2_vits_exp --num_prototypes 1000 --resume saved_models/CUB/dinov2_vits_exp_1000/9nopush0.8895.pth
python run_eval.py --base_architecture dinov2_vits_exp --num_prototypes 2000 --resume saved_models/CUB/dinov2_vits_exp_2000/9nopush0.8935.pth

# 9nopush0.9035.pth
python run_eval.py --base_architecture dinov2_vitb_exp --num_prototypes 1000 --resume saved_models/CUB/dinov2_vitb_exp_1000/14nopush0.9008.pth
# 9nopush0.9103.pth
python run_eval.py --base_architecture dinov2_vitb_exp --num_prototypes 2000 --resume saved_models/CUB/dinov2_vitb_exp_2000/16nopush0.9028.pth