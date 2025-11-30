#!/bin/bash

git submodule update --init --recursive

source set-env.sh

echo "" >> ~/.bashrc
cat set-env.sh >> ~/.bashrc

sudo apt update
sudo apt install -y imagemagick

git submodule update --init --recursive

python3 -m venv env
source env/bin/activate 

echo 
echo "Installing top-level deps"
pip install -qr requirements++.txt -r requirements+.txt -r requirements.txt

pushd submodules/Depth-Anything

python3 -m venv env
echo "Installing Depth-Anything deps"
source env/bin/activate

pip install -qr requirements.txt

popd

pushd submodules/pixel-perfect-depth

python3 -m venv env
source env/bin/activate

echo "Installing pixel perfect deps"
pip install -qr requirements.txt

# echo "Downloading pixel perfect checkpoint"
# PPD_CHECKPOINT_DIR=checkpoints
# mkdir -p $PPD_CHECKPOINT_DIR
# wget -P $PPD_CHECKPOINT_DIR "https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth"

# echo "Downloading pixel perfect depth anything v2 dependency checkpoint"
# wget -P $PPD_CHECKPOINT_DIR "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"

popd

pushd submodules/SharpDepth

python3 -m venv env
source env/bin/activate

echo "Installing SharpDepth deps"
pip install -qr docker/requirements.txt


if [ ! -d checkpoints ]; then
  echo "Fetching SharpDepth checkpoints"
  mkdir checkpoints
  cd checkpoints
  hf download andrew-healey/sharpdepth --include "sharpdepth-checkpoints/sharpdepth/**" --local-dir ckpts
  mkdir -p sharpdepth
  mv ckpts/sharpdepth-checkpoints/sharpdepth/* sharpdepth/ && rm -rf ckpts
fi

popd

pushd submodules/UniDepth

echo "Installing UniDepth in SharpDepth environment"

source ../SharpDepth/env/bin/activate

pip install -v -e . --no-deps

echo "Installing a custom op used by UniDepth. This will build some C++ code"
cd unidepth/ops/knn
python setup.py build install

popd

source env/bin/activate
pushd submodules/UniDepth

echo "Installing UniDepth in top-level environment"

pip install -v -e . --no-deps

echo "Installing a custom op used by UniDepth. This will build some C++ code"
cd unidepth/ops/knn
python setup.py build install

popd
