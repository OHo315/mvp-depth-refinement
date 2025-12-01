#!/bin/bash

git submodule update --init --recursive

python -m venv env

source env/bin/activate

source ./set-env.sh

echo "" >> ~/.bashrc
cat set-env.sh >> ~/.bashrc

echo "Installing top-level dependencies"

pip install -qr requirements++.txt -r requirements+.txt -r requirements.txt

pushd submodules/UniDepth

echo "Installing UniDepth in top-level environment"

pip install -v -e . --no-deps

echo "Installing a custom op used by UniDepth. This will build some C++ code"
cd unidepth/ops/knn
python setup.py build install

popd
