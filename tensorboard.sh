#!/bin/bash
NEWEST=$(ls -td models/*/ 2>/dev/null | head -1)
if [ -z "$NEWEST" ]; then
    echo "No folders found in models/"
    exit 1
fi
tensorboard --logdir="$NEWEST"
