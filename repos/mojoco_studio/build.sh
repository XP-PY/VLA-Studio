#!/bin/bash

PROGRAM_NAME=${1:-hello}

source "$(dirname "$0")/.env"

mkdir -p build

# Only the live viewer needs a window library. MuJoCo's bundled headers include
# SDL2; prefer its development package when available, otherwise use the Linux
# runtime soname. Existing headless exercises keep their original dependencies.
EXTRA_FLAGS=()
if [[ "$PROGRAM_NAME" == "robot_dynamics_viewer" ]]; then
    if pkg-config --exists sdl2 2>/dev/null; then
        read -r -a EXTRA_FLAGS <<< "$(pkg-config --cflags --libs sdl2)"
    else
        EXTRA_FLAGS=(-l:libSDL2-2.0.so.0)
    fi
fi

g++ -std=c++17 -O2 \
    -I"$MUJOCO_DIR/include" \
    -I"$MUJOCO_DIR/include/mujoco" \
    src/${PROGRAM_NAME}.cpp \
    -L"$MUJOCO_DIR/lib" \
    -Wl,-rpath,"$MUJOCO_DIR/lib" \
    -lmujoco "${EXTRA_FLAGS[@]}" \
    -o build/${PROGRAM_NAME}

# ./build/${PROGRAM_NAME}
