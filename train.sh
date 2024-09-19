#!/bin/bash

set -x

python main.py -arch densenet161
python main.py -arch dinov2_vitb14_reg4
python main.py -arch dinov2_vits14_reg4
