#!/bin/bash

source set-env.sh

sudo apt update
sudo apt install -y imagemagick

git submodule update --init --recursive

python3 -m venv env
source env/bin/activate 

echo 
echo "Installing top-level deps"
pip install -qr requirements++.txt -r requirements+.txt -r requirements.txt

pushd submodules/Depth-Anything

cat > .gitignore << 'EOF'
env/
checkpoints/
EOF


python3 -m venv env
echo "Installing Depth-Anything deps"
sed -i "/opencv-python/d" requirements.txt # Remove dependency which does not work in docker.
echo "" >> requirements.txt
echo "opencv-python-headless" >> requirements.txt # Replace dependency with something that does work in docker.
source env/bin/activate

pip install -qr requirements.txt

if [ ! -d checkpoints ]; then
  echo "Fetching depth anything checkpoints..."
  mkdir checkpoints
  wget -O checkpoints/depth_anything_metric_depth_indoor.pth "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt?download=true"
fi

popd

pushd submodules/pixel-perfect-depth

python3 -m venv env
source env/bin/activate

cat > .gitignore << 'EOF'
env/
checkpoints/
EOF

echo "Installing pixel perfect deps"
pip install -qr requirements.txt

if [ ! -d checkpoints ]; then
  echo "Fetching pixel perfect checkpoints..."
  mkdir checkpoints
  wget -O checkpoints/ppd.pth "https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth"
  wget -O checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
fi

popd

pushd submodules/SharpDepth

python3 -m venv env
source env/bin/activate

pip install -qr docker/requirements.txt

cat > .gitignore << 'EOF'
env/
checkpoints/
EOF

if [ ! -d checkpoints ]; then
  echo "Fetching SharpDepth checkpoints"
  mkdir checkpoints
  cd checkpoints
  wget https://github.com/Qualcomm-AI-research/SharpDepth/releases/download/v1.0/sharpdepth.tar.gz.part-aa
  wget https://github.com/Qualcomm-AI-research/SharpDepth/releases/download/v1.0/sharpdepth.tar.gz.part-ab 
  wget https://github.com/Qualcomm-AI-research/SharpDepth/releases/download/v1.0/sharpdepth.tar.gz.part-ac
  cat sharpdepth.tar.gz.part-* >sharpdepth.tar.gz
  tar zxvf sharpdepth.tar.gz
  rm sharpdepth.tar.gz*
fi

popd

pushd submodules/UniDepth

echo "Installing UniDepth"

source ../SharpDepth/env/bin/activate

pip install -v -e . --no-deps

echo "Installing a custom op used by UniDepth. This will build some C++ code"
cd unidepth/ops/knn
python setup.py build install

popd