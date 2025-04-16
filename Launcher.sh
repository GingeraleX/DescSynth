#!/bin/bash

echo "🎬 Descriptor Synth Launcher"
echo ""

# Check for Python 3.10+
PYTHON=$(command -v python3 || command -v python)
if [[ -z "$PYTHON" ]]; then
    echo "❌ Python not found. Please install Python 3.10+"
    exit 1
fi

PYVER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VER="3.10"

vercomp () {
    if [[ "$1" == "$2" ]]; then return 0; fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do ver1[i]=0; done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then ver2[i]=0; fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then return 1; fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then return 0; fi
    done
    return 0
}

if ! vercomp "$PYVER" "$REQUIRED_VER"; then
    echo "❌ Python $PYVER is too old. Please install Python $REQUIRED_VER or newer."
    exit 1
fi

echo "✓ Python $PYVER found."

# Install requirements if needed
echo "Installing required Python packages..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install numpy scipy matplotlib librosa soundfile sounddevice

# Run the main Python script
echo ""
echo "🚀 Running Descriptor Synth..."
$PYTHON script.py
