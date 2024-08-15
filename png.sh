#!/bin/zsh

set -ex
cargo r -- models/mnist-12.onnx
dot -Tpng graph.dot -o a.png
