#!/bin/bash

# === Style ===
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'

echo -e "\n${GRN}Descriptor Synth Launcher${NC}"
echo "--------------------------------------------"

REQUIRED="3.10"
PY=$(command -v python3.10 || command -v python3 || command -v python)

# --- Python Check ---
if [[ -z "$PY" ]]; then
    echo -e "${RED}Python not found.${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &>/dev/null; then
            echo -e "${YLW}Installing Python via Homebrew...${NC}"
            brew install python@3.10
        else
            echo -e "${RED}Homebrew not found.${NC}"
            echo "Please install Python 3.10 manually from: https://www.python.org/downloads/mac-osx/"
            exit 1
        fi
    else
        echo -e "${YLW}Installing Python via APT...${NC}"
        sudo apt update && sudo apt install -y python3.10 python3.10-venv python3.10-dev
    fi
    PY=$(command -v python3.10)
fi

VERSION=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(printf '%s\n' "$REQUIRED" "$VERSION" | sort -V | head -n1)" != "$REQUIRED" ]]; then
    echo -e "${RED}Found Python $VERSION — 3.10+ required.${NC}"
    exit 1
fi
echo -e "${GRN}Python $VERSION OK${NC}"

# --- pip check ---
if ! $PY -m pip --version &>/dev/null; then
    echo -e "${YLW}Installing pip...${NC}"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PY get-pip.py && rm get-pip.py
fi

echo -e "${GRN}pip available${NC}"

# --- Dependency check ---
DEPS=(numpy scipy matplotlib librosa soundfile sounddevice)
MISSING=()
for pkg in "${DEPS[@]}"; do
    $PY -c "import $pkg" &>/dev/null || MISSING+=($pkg)
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo -e "${YLW}Installing missing packages: ${MISSING[*]}${NC}"
    $PY -m pip install --quiet "${MISSING[@]}"
else
    echo -e "${GRN}All dependencies already satisfied${NC}"
fi

# --- Run GUI ---
GUI_PATH="./Code/GUI.py"
if [[ ! -f "$GUI_PATH" ]]; then
    echo -e "${RED}GUI.py not found in ./Code${NC}"
    exit 1
fi

echo -e "${GRN}Launching GUI...${NC}\n"
$PY "$GUI_PATH"
