#!/bin/bash

# --- Configuration ---
# The delay time
DELAY="4h"

# The specific Python executable for your 'act' environment
PYTHON_BIN="/home/pengtao/micromamba/envs/act/bin/python3"

# The script you want to run
SCRIPT_NAME="optimize.py"
LOG_NAME="optimize.log"

# --- Automatic Path Detection ---
# Get the directory where this script is running
WORK_DIR=$(pwd)

echo "----------------------------------------"
echo "Scheduling Task (Micromamba Environment)"
echo "  Interpreter: $PYTHON_BIN"
echo "  Script:      $WORK_DIR/$SCRIPT_NAME"
echo "  Delay:       $DELAY"
echo "  Log File:    $WORK_DIR/$LOG_NAME"
echo "----------------------------------------"

# --- Verify Paths Exist ---
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python executable not found at $PYTHON_BIN"
    exit 1
fi

if [ ! -f "$WORK_DIR/$SCRIPT_NAME" ]; then
    echo "Error: Script not found at $WORK_DIR/$SCRIPT_NAME"
    exit 1
fi

# --- The Payload Command ---
# We use the absolute path to python, which automatically uses the 
# libraries installed in the 'act' environment.
CMD_STRING="$PYTHON_BIN $WORK_DIR/$SCRIPT_NAME > $WORK_DIR/$LOG_NAME 2>&1"

# --- Execute Systemd Run ---
# We wrap it in /bin/bash -c so the redirect (>) works correctly
systemd-run --user --on-active=${DELAY} /bin/bash -c "$CMD_STRING"

echo "Success! Timer started."
echo "Check status with: systemctl --user list-timers"