#!/bin/bash

git submodule update --init --recursive

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
source env/bin/activate

pip install -qr requirements.txt

popd

pushd submodules/pixel-perfect-depth

python3 -m venv env
source env/bin/activate

echo "Installing pixel perfect deps"
pip install -qr requirements.txt

popd

pushd submodules/SharpDepth

python3 -m venv env
source env/bin/activate

pip install -qr docker/requirements.txt

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