#!/bin/bash

echo ""
echo "🎬 Launching Descriptor Synth"
echo "----------------------------"
echo ""

# Detect OS and choose python command
if [[ "$OS" == "Windows_NT" ]]; then
    PYTHON=py
else
    PYTHON=$(command -v python3 || command -v python)
fi

# Check if Python is available
if ! command -v $PYTHON &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.10+."
    read -p "Press Enter to exit..."
    exit 1
fi

# Check version
PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VER="3.10"

ver_ok=$($PYTHON -c "import sys; print(sys.version_info >= (3, 10))")
if [[ "$ver_ok" != "True" ]]; then
    echo "❌ Found Python $PYVER — need version 3.10 or higher."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✓ Python $PYVER found."

# Install required packages
echo ""
echo "🔍 Installing Python packages..."
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet numpy scipy matplotlib librosa soundfile

# Run your script
if [[ ! -f "script.py" ]]; then
    echo "❌ script.py not found in the current folder."
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "🚀 Running script.py..."
$PYTHON script.py

status=$?
if [[ $status -ne 0 ]]; then
    echo ""
    echo "❌ script.py exited with error code $status"
    read -p "Press Enter to exit..."
    exit $status
fi

echo ""
echo "✅ Done!"
read -p "Press Enter to close..."
