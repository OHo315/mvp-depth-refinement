#!/bin/bash
pip install unzip

bash ./script/data_fetch/hypersim-hf.sh &
bash ./script/data_fetch/nyu.sh &
bash ./script/data_fetch/middlebury.sh &
bash ./script/data_fetch/arkitscenes-hf.sh &
bash ./script/data_fetch/waymo.sh & 
bash ./script/data_fetch/diode-indoor.sh &
echo "Waiting for all datasets to be fetched..." &
wait
