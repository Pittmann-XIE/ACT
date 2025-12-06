#!/bin/bash

# --- Configuration ---
PYTHON_BIN="/home/pengtao/micromamba/envs/act/bin/python3"
BLOCKER_SCRIPT="gpu_blocker.py"
REAL_JOB_SCRIPT="optimize.py"
REAL_JOB_LOG="optimize.log"
WAIT_TIME="4h"

# Get current directory
WORK_DIR=$(pwd)

echo "=============================================="
echo "Pipeline Started at: $(date)"
echo "Mode: Reserve GPU for $WAIT_TIME, then execute job."
echo "=============================================="

# 1. Start the GPU Blocker in the background
echo "[Phase 1] Starting GPU Blocker..."
$PYTHON_BIN $WORK_DIR/$BLOCKER_SCRIPT > $WORK_DIR/blocker_status.log 2>&1 &

# 2. Capture the Process ID (PID) of the blocker
BLOCKER_PID=$!
echo "GPU Blocker running with PID: $BLOCKER_PID"

# 3. Wait for the specified time
echo "[Phase 2] Sleeping for $WAIT_TIME..."
sleep $WAIT_TIME

# 4. Time is up! Kill the blocker to release GPU memory
echo "[Phase 3] Time reached. Stopping GPU Blocker..."
if ps -p $BLOCKER_PID > /dev/null; then
    kill $BLOCKER_PID
    echo "Blocker (PID $BLOCKER_PID) killed. GPU memory released."
else
    echo "Warning: Blocker process was not found. It might have crashed earlier."
fi

# 5. Brief pause to ensure OS cleans up GPU resources
sleep 5

# 6. Run the Real Job
echo "[Phase 4] Starting Real Job ($REAL_JOB_SCRIPT)..."
$PYTHON_BIN $WORK_DIR/$REAL_JOB_SCRIPT > $WORK_DIR/$REAL_JOB_LOG 2>&1

echo "=============================================="
echo "Pipeline Finished at: $(date)"
echo "=============================================="