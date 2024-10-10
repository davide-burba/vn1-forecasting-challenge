#!/bin/bash

# Check if the folder argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder> <phase>"
  exit 1
fi

# Check if the phase argument is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <folder> <phase>"
  exit 1
fi

# Get the folder and phase from the arguments
FOLDER=$1
PHASE=$2

# Loop over all YAML files in the folder
for FILE in "$FOLDER"/*.yaml; do
  if [ -f "$FILE" ]; then
    echo "Processing $FILE"
    uv run python scripts/run_phase_$PHASE.py --config_path $FILE
  else
    echo "No YAML files found in $FOLDER"
  fi
done