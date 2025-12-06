#!/bin/bash

# 1. Define paths
PYTHON_BIN="/home/pengtao/micromamba/envs/act/bin/python3"
SCRIPT_NAME="optimize.py"
LOG_NAME="optimize.log"
WORK_DIR=$(pwd)

# 2. Run the command immediately
# The GPU blocker has already finished waiting, so we start now.
echo "Starting optimization task..."
$PYTHON_BIN $WORK_DIR/$SCRIPT_NAME > $WORK_DIR/$LOG_NAME 2>&1