#!/bin/bash
# This scripts acts as a wrapper around the main program, allowing us to save any proper data before running.
echo "Initialising display ..."
echo "Display initialised"

echo "Starting application ..."
cd rlcw
python3 -m main
