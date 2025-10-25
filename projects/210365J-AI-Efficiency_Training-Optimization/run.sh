#!/bin/bash
# Script to run the main.py with the correct Python environment

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the src directory
cd "$DIR/src"

# Run main.py with the virtual environment Python
"$DIR/.venv/bin/python" main.py "$@"
