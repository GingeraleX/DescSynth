# 🎵 Descriptor Synth

**Descriptor Synth** is a Python-based audio analysis and synthesis tool. It extracts temporal and spectral descriptors from a `.wav` file and uses them to control a synthesis engine via a graphical interface.

---

## 🚀 Features

- Descriptor extraction (ADSR, spectral centroid, skewness, flatness, etc.)
- GUI with sliders for real-time descriptor manipulation
- Curve editor for time-varying descriptor visualization and editing
- Additive synthesis engine with noise and spectral shaping
- Live playback and export to `.wav`

---

## 📦 Requirements

- Python 3.10 or higher  
- Dependencies:
  ```
  pip install numpy scipy matplotlib librosa soundfile sounddevice
  ```

> On Windows, use the `py` launcher if `python` isn't recognized.

---

## 🖥 Installation

1. Clone or download this repository.
2. Place your audio file (e.g., `input.wav`) in the same folder.
3. Install requirements:
   ```bash
   pip install numpy scipy matplotlib librosa soundfile sounddevice
   ```

---

## ▶️ Usage

Run the main script:

```bash
python script.py
```

or on Windows:

```bash
py script.py
```

---

## 🎛 GUI Overview

- **Load WAV**: Load your audio file.
- **Synthesize & Play**: Analyze and synthesize based on current settings.
- **Save As...**: Export the synthesized audio.
- **Sliders**: Adjust descriptor weights for synthesis.
- **Curve Editor**: Click any descriptor thumbnail to open an editable plot.

---

## 🧠 How It Works

1. Extracts descriptors from the input audio using STFT and custom functions.
2. Maps these descriptors to synthesis parameters.
3. Generates new sound via additive synthesis + noise + envelope.
4. Provides real-time control and preview of changes through the GUI.

---

## 📝 Modifications

To change the input file, edit the script or load a file from the GUI.

To customize descriptor mappings or synthesis logic, look into:
- `extract_descriptors(...)`
- `synthesize(...)`

---

## 🐞 Troubleshooting

- ❌ **"Python not found"**: Use `py` on Windows or install Python from [python.org](https://www.python.org/downloads/)
- ❌ **"No module named librosa"**: Run `pip install librosa`
- ❌ **No output or sound**: Check your audio device and descriptor values

---

## 📃 License

Commercial License – feel free to contact me for any info.