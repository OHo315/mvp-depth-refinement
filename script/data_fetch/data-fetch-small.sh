#!/bin/bash
pip install unzip

bash ./script/data_fetch/hypersim-small.sh
bash ./script/data_fetch/nyu.sh
bash ./script/data_fetch/middlebury.sh
bash ./script/data_fetch/arkitscenes-hf-small.sh
bash ./script/data_fetch/waymo-small.sh
bash ./script/data_fetch/diode-indoor-small.sh

