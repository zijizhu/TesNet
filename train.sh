#!/bin/bash

set -x

python main.py -arch dinov2_vitb_exp -num_prototypes 1000
python main.py -arch dinov2_vitb_exp -num_prototypes 2000
python main.py -arch dinov2_vits_exp -num_prototypes 1000
python main.py -arch dinov2_vits_exp -num_prototypes 2000
