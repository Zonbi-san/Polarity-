#!/bin/sh

git clone https://gitlab.com/conradsnicta/armadillo-code.git

cd armadillo-code || exit

mkdir -p build && cd build || exit

cmake ..

make -j 8

sudo make install