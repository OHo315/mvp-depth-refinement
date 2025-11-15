#!/bin/bash

# Start the SSH server in the background
# /usr/sbin/sshd -D -e &

source ./set-env.sh

echo "Started Running..."

echo "step 1: setup and fetch repos"
bash ./setup.sh

source env/bin/activate

echo "step 2: dataset fetch"
bash ./script/data_fetch/data-fetch.sh

echo "set 3: depth anything"
bash ./script/external_models/run-depth-anything.sh

# echo "step 4: ppd"
# bash ./script/external_models/run-ppd.sh

# Setup vast cli tool
CONTAINER_API_KEY="$(cat ~/.vast_api_key)"
CONTAINER_ID="$(cat ~/.vast_container_label)"

vastai set api-key $CONTAINER_API_KEY 

vastai stop instance $CONTAINER_ID

