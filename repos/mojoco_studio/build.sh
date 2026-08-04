#!/bin/bash

PROGRAM_NAME=${1:-hello}

source "$(dirname "$0")/.env"

mkdir -p build

g++ -std=c++17 -O2 \
    -I"$MUJOCO_DIR/include" \
    -I"$MUJOCO_DIR/include/mujoco" \
    src/${PROGRAM_NAME}.cpp \
    -L"$MUJOCO_DIR/lib" \
    -Wl,-rpath,"$MUJOCO_DIR/lib" \
    -lmujoco \
    -o build/${PROGRAM_NAME}

# ./build/${PROGRAM_NAME}