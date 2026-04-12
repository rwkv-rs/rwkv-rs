#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  git \
  cmake \
  ninja-build \
  protobuf-compiler \
  libcap-ng-dev \
  libdrm-dev \
  libssl-dev

echo "== tool checks =="
command -v cc
command -v c++
command -v cmake
command -v ninja || true
command -v protoc
command -v pkg-config
command -v git

echo "== versions =="
cc --version | head -n 1
c++ --version | head -n 1
cmake --version | head -n 1
protoc --version
pkg-config --version

echo "== pkg-config checks =="
pkg-config --libs libdrm
pkg-config --libs libcap-ng

echo "build environment looks ready."
