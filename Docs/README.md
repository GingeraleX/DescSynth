# Descriptor-Controlled Audio Synthesis

**Descriptor-Controlled Synthesis** is a creative sound tool that transforms the sonic qualities of an existing sound into controls for generating new audio. You load a sound, extract its unique "descriptors," and use them to guide synthesis in a visually rich, GUI-based playground.

Designed with musicians and sound designers in mind, this tool is for exploring timbre through descriptors — no programming required.

---

## 🎛 Features

- 🎨 Easy-to-use **Graphical Interface** for tweaking and sculpting sound
- 🔎 Intelligent **descriptor extraction** (e.g., attack, spectral centroid, noisiness)
- 🧠 Custom **curve editing** for time-varying control
- 🎚 Interactive **parameter sliders** grouped by sound properties
- 🔊 Built-in audio playback and waveform/spectrogram preview
- 💾 Save and load your own descriptor curves

---

## 📦 Download & Installation

### 👶 For Beginners (No Git, No Terminal)

1. Go to the [GitHub Releases](https://github.com/yourusername/descriptor-synthesis/releases) page *(or wherever the .zip is hosted)*.
2. Download the latest `.zip` file.
3. Right-click → **Extract All** (Windows) or double-click the zip to extract (macOS).
4. Inside the folder, double-click:
   - `Launcher.bat` for **Windows**
   - or run `Launcher.sh` on **macOS/Linux**

This will automatically:
- Check/install Python 3.10
- Install needed libraries
- Open the GUI for you

### 🧑‍💻 For Developers (Git Method)

```bash
git clone https://github.com/yourusername/descriptor-synthesis.git
cd descriptor-synthesis
./Launcher.sh  # or double-click Launcher.bat on Windows
```

---

## 🖥 How to Use

1. Click the 📂 button to **load a sound file** (.wav or .mp3).
2. The tool extracts **descriptors** like envelope shape, energy, pitch, and spectral features.
3. Use the sliders and visual curves to tweak each parameter.
4. Click ▶️ to **generate and hear** your custom sound.
5. Optionally, save descriptor sets or load new ones.

---

## 📁 Folder Structure

```
descriptor-synthesis/
├── Code/
│   ├── GUI.py
│   ├── Extractor.py
│   ├── Engine.py
│   ├── Curve.py
├── Inputs/                # Put audio files here to work with
├── Outputs/               # Synthesized results and saved descriptor data
├── Docs/
│   ├── README.md          # This file
│   ├── goal.pdf           # Project concept overview
│   └── references/        # Research and citations
├── Launcher.sh            # Startup script for macOS/Linux
├── Launcher.bat           # Startup for Windows
└── LICENSE                # License info
```

---

## 📄 License

This project is licensed under the terms of the [Commercial License](LICENSE).

---

## 💬 Feedback & Contributions

Are you a sound designer or musician with ideas? Found a bug? Want to improve the GUI?  
Feel free to open an issue or submit a pull request — your input is welcome!

---

Happy synthesizing! 🎶
