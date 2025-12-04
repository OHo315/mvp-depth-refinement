#!/bin/sh

wget -qO csvlens.tar.xz https://github.com/YS-L/csvlens/releases/latest/download/csvlens-x86_64-unknown-linux-gnu.tar.xz
mkdir csvlens-temp
tar xf csvlens.tar.xz --strip-components=1 -C csvlens-temp
sudo mv csvlens-temp/csvlens /usr/local/bin/
rm -rf csvlens.tar.xz csvlens-temp

