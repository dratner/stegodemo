#!/bin/bash

# Exit on error
set -e

echo "=== StegoGPT Setup Script ==="
echo "This script will set up the StegoGPT C++ implementation."

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

echo "=== Setup Complete ==="
echo "You can now run the application with:"
echo "./bin/stego_app 2"
echo "Where 2 is the number of bits per token."