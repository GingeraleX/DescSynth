import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Engine import synthesize
from Curve import CurveEditor
from Extractor import extract_descriptors_full
import sounddevice as sd
import librosa.display
from tkinter import filedialog
import pickle
from scipy.io import wavfile

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.bind("<space>", lambda e: self.play_audio())

        self.root.title("Descriptor-Controlled Synthesis")
        
        # Toolbar at top
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side="top", fill="x", padx=5, pady=3)
        
        ttk.Button(self.toolbar, text="📂", command=self.load_and_extract_descriptors).pack(side="left", padx=4)
        ttk.Button(self.toolbar, text="▶️", command=self.play_audio).pack(side="left", padx=4)

        self.params = {}
        self.controls = []
        self.descriptor_data = {}

        self.notebook = ttk.Notebook(self.root)
        self.tabs = {
            "Temporal/Envelope": ttk.Frame(self.notebook),
            "Energy": ttk.Frame(self.notebook),
            "Spectral": ttk.Frame(self.notebook),
            "Harmonic/Freq": ttk.Frame(self.notebook),
            "System": ttk.Frame(self.notebook),
            "Curves": ttk.Frame(self.notebook)
        }

        for name, frame in self.tabs.items():
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=3)
            self.notebook.add(frame, text=name)
            
        self.notebook.pack(expand=1, fill='both')

        self.time_varying_keys = [
            "FrameErg", "ZCR", "SpecCent", "SpecSpread", "SpecSkew", "SpecKurt",
            "SpecDecr", "SpecFlat", "SpecCrest", "SpecSlope", "SpecVar",
            "SpectroTempVar", "SpecShape", "F0", "HarmErg", "NoiseErg",
            "Noisiness", "InHarm", "HarmDev", "OddEveRatio"
        ]

        self.descriptor_groups = {
            "Temporal/Envelope": [
                ("Attack", 0.001, 1.0, 0.001),
                ("Decay", 0.001, 1.0, 0.001),
                ("Sustain", 0.0, 1.0, 0.01),
                ("Release", 0.001, 2.0, 0.001),
                ("Temporal Centroid Strength", 0.0, 2.0, 0.01),
                ("Autocorrelation Strength", 0.0, 2.0, 0.01),
                ("Effective Duration Strength", 0.0, 2.0, 0.01),
            ],
            "Energy": [
                ("Frame Energy Strength", 0.0, 2.0, 0.01),
                ("Energy Modulation", 0.0, 1.0, 0.01),
                ("Frequency of Energy Modulation Strength", 0.1, 20.0, 0.1),
                ("Zero Crossing Rate Strength", 0.0, 2.0, 0.01),
                ("Output Gain", 0.0, 2.0, 0.01),
            ],
            "Spectral": [
                ("Spectral Centroid Strength", 0.0, 2.0, 0.01),
                ("Spectral Spread Strength", 0.0, 2.0, 0.01),
                ("Spectral Skewness Strength", -2.0, 2.0, 0.01),
                ("Spectral Kurtosis Strength", 0.0, 2.0, 0.01),
                ("Spectral Decrease Strength", 0.0, 2.0, 0.01),
                ("Spectral Flatness Strength", 0.0, 2.0, 0.01),
                ("Spectral Crest Strength", 0.0, 2.0, 0.01),
                ("Spectral Slope Strength", 0.0, 2.0, 0.01),
                ("Spectral Variation Strength", 0.0, 2.0, 0.01),
                ("Spectro-temporal Variation Strength", 0.0, 2.0, 0.01),
                ("Spectral Shape Strength", 0.0, 2.0, 0.01),
                ("Spectral Flux Strength", 0.0, 2.0, 0.01),
                ("Spectral Shape Morph Strength", 0.0, 2.0, 0.01),
            ],
            "Harmonic/Freq": [
                ("Fundamental Frequency Strength", 0.0, 2.0, 0.01),
                ("Harmonic Energy Strength", 0.0, 2.0, 0.01),
                ("Noise Energy Strength", 0.0, 2.0, 0.01),
                ("Noisiness Strength", 0.0, 2.0, 0.01),
                ("Inharmonicity Strength", 0.0, 2.0, 0.01),
                ("Harmonic Spectral Deviation", 0.0, 1.0, 0.01),
                ("Odd to Even Harmonic Ratio Strength", 0.0, 2.0, 0.01),
            ],
            "System": [
                ("Amp Decay", 0.5, 5.0, 0.1),
                ("Tremolo Depth", 0.0, 1.0, 0.01),
                ("Sample Rate", 8000, 48000, 1000),
                ("Block Size", 256, 4096, 64),
                ("Hop Size", 128, 2048, 64),
                ("Number of Partials", 1, 64, 1),
                ("Descriptor Smoothing", 0.0, 1.0, 0.01),
            ]
        }

        self.default_values = {
            "Sample Rate": 44100,
            "Block Size": 1024,
            "Hop Size": 512,
            "Duration (s)": 2.0,
            "Amp Decay": 1.2,
            "Tremolo Depth": 0.0,
            # Add more if needed
        }

        self.options = {
            "Tremolo Shape": ["sine", "triangle", "square"],
            "Window Type": ["hann", "hamming", "blackman", "rect"],
            "Envelope Type": ["adsr", "gaussian", "skewed", "exp"],
            "Noise Color": ["white", "pink", "brown"],
            "Filter Type": ["bandpass", "lowpass", "highpass"]
        }

        self.option_vars = {}

        for tab, controls in self.descriptor_groups.items():
            for i, (label, min_val, max_val, step) in enumerate(controls):
                self.add_slider(self.tabs[tab], label, min_val, max_val, step, i)

        for i, (label, choices) in enumerate(self.options.items()):
            self.add_dropdown(self.tabs["System"], label, choices, i + len(self.descriptor_groups["System"]))

        self.curve_panel = ttk.LabelFrame(self.tabs["Curves"], text="Time-Varying Descriptor Curves")
        self.curve_panel.pack(fill='both', expand=True, padx=10, pady=10)

        self.canvas_container = tk.Canvas(self.curve_panel)
        self.scrollbar = tk.Scrollbar(self.curve_panel, orient="vertical", command=self.canvas_container.yview)
        self.scroll_frame = ttk.Frame(self.canvas_container)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas_container.configure(scrollregion=self.canvas_container.bbox("all")))

        self.canvas_container.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        self.canvas_container.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_container.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def load_and_extract_descriptors(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            descriptors = extract_descriptors_full(file_path)
            self.descriptor_data.update(descriptors)
            self.refresh_curve_thumbnails()
            self.populate_gui_with_descriptors(descriptors)
            self.plot_waveform_and_spectrogram(file_path)

    def plot_waveform_and_spectrogram(self, file_path):
        y, sr = librosa.load(file_path, sr=None)

        preview_win = tk.Toplevel(self.root)
        preview_win.title("Waveform & Spectrogram Preview")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Waveform")

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title("Spectrogram (dB)")

        fig.colorbar(img, ax=ax2, format="%+2.0f dB")
        fig.tight_layout()

        for ax in (ax1, ax2):
            ax.tick_params(labelsize=8)

        canvas = FigureCanvasTkAgg(fig, master=preview_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)
        ttk.Button(preview_win, text="Close", command=preview_win.destroy).pack(pady=5)

    def populate_gui_with_descriptors(self, descriptors):
        key_map = {
            "Att": "Attack",
            "Dec": "Decay",
            "SusLevel": "Sustain",
            "Rel": "Release",
            "TempCent": "Temporal Centroid Strength",
            "SusTime": "Effective Duration Strength",
            "FreqMod": "Frequency of Energy Modulation Strength",
            "ZCR": "Zero Crossing Rate Strength",
            "SpecCent": "Spectral Centroid Strength",
            "SpecSpread": "Spectral Spread Strength",
            "SpecSkew": "Spectral Skewness Strength",
            "SpecKurt": "Spectral Kurtosis Strength",
            "SpecDecr": "Spectral Decrease Strength",
            "SpecFlat": "Spectral Flatness Strength",
            "SpecCrest": "Spectral Crest Strength",
            "SpecSlope": "Spectral Slope Strength",
            "SpecVar": "Spectral Variation Strength",
            "SpectroTempVar": "Spectro-temporal Variation Strength",
            "SpecShape": "Spectral Shape Strength",
            "F0": "Fundamental Frequency Strength",
            "HarmErg": "Harmonic Energy Strength",
            "NoiseErg": "Noise Energy Strength",
            "Noisiness": "Noisiness Strength",
            "InHarm": "Inharmonicity Strength",
            "OddEveRatio": "Odd to Even Harmonic Ratio Strength",
            "FrameErg": "Frame Energy Strength",
            "Autocorrelation": "Autocorrelation Strength"
        }

        for desc_key, gui_label in key_map.items():
            if gui_label in self.params and desc_key in descriptors:
                val = descriptors[desc_key]
                if isinstance(val, (float, int)):
                    self.params[gui_label].set(val)
                elif isinstance(val, np.ndarray):
                    continue

    def open_curve_editor(self, key):
        CurveEditor(
            master=self.root,
            curve_data=self.descriptor_data.get(key, np.ones(100)),  # fallback to dummy
            key=key,
            apply_callback=self.update_curve
        )

    def update_curve(self, key, curve):
        self.descriptor_data[key] = curve
        self.refresh_curve_thumbnails()

    def add_slider(self, tab, label, min_val, max_val, step, row):
        ttk.Label(tab, text=label).grid(row=row, column=0, sticky='w')
        default_val = self.default_values.get(label, (min_val + max_val) / 2)
        var = tk.DoubleVar(value=default_val)
        scale = tk.Scale(tab, from_=min_val, to=max_val, resolution=step, variable=var,
                         orient='horizontal', length=300, showvalue=True)
        scale.grid(row=row, column=1, sticky='ew', padx=5)
        self.params[label] = var
        self.controls.append(scale)

    def add_dropdown(self, tab, label, choices, row):
        ttk.Label(tab, text=label).grid(row=row, column=0, sticky='w')
        var = tk.StringVar(value=choices[0])
        dropdown = ttk.OptionMenu(tab, var, choices[0], *choices)
        dropdown.grid(row=row, column=1, sticky='ew', padx=5)
        self.option_vars[label] = var

    def load_curve_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            with open(file_path, 'rb') as f:
                self.descriptor_data = pickle.load(f)
            self.refresh_curve_thumbnails()

    def refresh_curve_thumbnails(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        for idx, key in enumerate(self.time_varying_keys):
            if key not in self.descriptor_data:
                continue
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(self.descriptor_data[key])
            ax.set_title(key)
            ax.axis('off')
            canvas = FigureCanvasTkAgg(fig, master=self.scroll_frame)
            fig.tight_layout()
            canvas.draw()
            plt.close(fig)
            widget = canvas.get_tk_widget()
            widget.grid(row=idx // 4, column=idx % 4, padx=5, pady=5)
            widget.bind("<Button-1>", lambda e, k=key: self.open_curve_editor(k))

    def play_audio(self):
        self.render_audio()
        try:
            sr, audio = wavfile.read("descriptor_synthesis_gui.wav")
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / np.max(np.abs(audio))
            sd.play(audio, samplerate=sr)
        except Exception as e:
            print(f"Failed to play audio: {e}")
    def render_audio(self):
        print("Rendering with current descriptor values...")
        param_dict = {key: var.get() for key, var in self.params.items()}
        param_dict.update({key: var.get() for key, var in self.option_vars.items()})
        audio = synthesize(param_dict, self.descriptor_data)
        filename = "descriptor_synthesis_gui.wav"
        sf.write(filename, audio, int(param_dict["Sample Rate"]))
        print(f"Audio rendering complete. Saved as {filename}.")

if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
