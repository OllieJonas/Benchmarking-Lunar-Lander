#!/bin/bash

# regex: ^[A-Za-z]+ - ([0-9]{2}-){2}[0-9]{2}_([0-9]{2}-){2}[0-9]{4}

echo "Starting application..."
cd rlcw
time python3 -m main
