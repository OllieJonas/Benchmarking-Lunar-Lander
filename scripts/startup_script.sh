#!/bin/bash
# This scripts acts as a wrapper around the main program, allowing us to save any proper data before running.
echo "Starting application ..."
cd rlcw/notebooks
jupyter notebook --allow-root
