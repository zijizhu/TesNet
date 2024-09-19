#!/bin/bash

set -x

python main.py -arch dinov2_vitb_exp
python main.py -arch dinov2_vits_exp
