#!/bin/bash

# Define the delay (3 hours)
# 3 hours * 60 mins * 60 secs = 10800 seconds
DELAY="2h"

echo "Waiting $DELAY before starting the training..."
sleep $DELAY

echo "Starting training now: $(date)"

# Execute your command
# -u ensures output is unbuffered so the log updates in real-time
python3 -u train_joint_1_step.py > train_joint_1_step_realsense_aria_100.log 2>&1 &

echo "Process started in background with PID $!"