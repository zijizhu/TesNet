#!/bin/bash

set -x

python main.py -arch dinov2_vitb_exp -dataset Cars -num_prototypes 980
python main.py -arch dinov2_vits_exp -dataset Cars -num_prototypes 980
python main.py -arch dinov2_vitb_exp -dataset Cars -num_prototypes 588
python main.py -arch dinov2_vits_exp -dataset Cars -num_prototypes 588

python main.py -arch dinov2_vitb_exp -dataset Dogs -num_prototypes 600
python main.py -arch dinov2_vits_exp -dataset Dogs -num_prototypes 600
python main.py -arch dinov2_vitb_exp -dataset Dogs -num_prototypes 360
python main.py -arch dinov2_vits_exp -dataset Dogs -num_prototypes 360

# python main.py -arch dino_vitb16 -num_prototypes 600
# python main.py -arch dino_vitb16 -num_prototypes 1000
# python main.py -arch dino_vitb16 -num_prototypes 2000
# python main.py -arch dinov2_vitb_exp -num_prototypes 600
# python main.py -arch dinov2_vits_exp -num_prototypes 600

# python main.py -arch dinov2_vitb_exp -num_prototypes 1000
# python main.py -arch dinov2_vitb_exp -num_prototypes 2000
# python main.py -arch dinov2_vits_exp -num_prototypes 1000
# python main.py -arch dinov2_vits_exp -num_prototypes 2000
