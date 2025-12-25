#!/bin/bash

mkdir -p results

nvcc -o tiled tiled.cu || exit 1
nvcc -o naive naive.cu || exit 1

for N in 256 512 1024 2048 4096 8192 16384; do
    ./tiled $N >> results/tiled.txt
    ./naive $N >> results/naive.txt
done

rm tiled naive
