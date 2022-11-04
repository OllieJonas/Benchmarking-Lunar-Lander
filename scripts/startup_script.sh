#!/bin/bash
# This scripts acts as a wrapper around the main program, allowing us to save any proper data before running.
echo "Initialising display ..."
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset
export DISPLAY=:1
echo "Display initialised"

echo "Starting application ..."
cd rlcw
python3 -m main