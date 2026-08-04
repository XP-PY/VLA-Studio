#!/bin/bash

XML_NAME=${1:-hello}

source "$(dirname "$0")/.env"

"$MUJOCO_DIR/bin/simulate" assets/${XML_NAME}.xml