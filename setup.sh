#!/bin/bash

source set-env.sh

git submodule update --init --recursive

python3 -m venv env
source env/bin/activate 

cat > .gitignore << 'EOF'
env/
checkpoints/
EOF

echo "Installing top-level deps"
pip install -qr requirements++.txt -r requirements+.txt -r requirements.txt

pushd submodules/Depth-Anything
python3 -m venv env
echo "Installing Depth-Anything deps"
sed -i "/opencv-python/d" requirements.txt # Remove dependency which does not work in docker.
echo "\nopencv-python-headless" >> requirements.txt # Replace dependency with something that does work in docker.
source env/bin/activate

pip install -qr requirements.txt

echo "Fetching depth anything checkpoints..."
mkdir checkpoints
wget -O checkpoints/depth_anything_metric_depth_indoor.pth "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt?download=true"

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

echo "Fetching pixel perfect checkpoints..."
mkdir checkpoints
wget -O checkpoints/ppd.pth "https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth"
wget -O checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

popd
