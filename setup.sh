#!/bin/bash

git submodule update --init --recursive

python -m venv env

source env/bin/activate

source ./set-env.sh

echo "" >> ~/.bashrc
cat set-env.sh >> ~/.bashrc

mkdir -p $BASE_DATA_DIR

echo "Installing top-level dependencies..."

pip install -qr requirements++.txt -r requirements+.txt -r requirements.txt

echo "Done installing top-level dependencies"

pushd submodules/UniDepth > /dev/null

if ! python -c "import unidepth" &>/dev/null; then
  echo "Installing UniDepth in top-level environment"
  pip install -v -e . --no-deps
fi

if ! python -c "import torch; import KNN" &>/dev/null; then
  echo "Installing a custom op used by UniDepth. This will build some C++ code"
  cd unidepth/ops/knn
  python setup.py build install
fi

popd > /dev/null
