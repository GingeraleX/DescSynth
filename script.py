# descriptor_synth_system.py (Full Version)
# Complete pipeline: Analysis -> Descriptor Mapping -> Synthesis -> GUI

import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial

# -------------------------------------------------
# === Descriptor Extraction ===
# -------------------------------------------------
def extract_descriptors(file_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    time = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)

    def frame_energy(m):
        return np.sum(m**2, axis=0)

    def centroid(m):
        return np.sum(freqs[:, None] * m, axis=0) / (np.sum(m, axis=0) + 1e-10)

    def spread(m, c):
        return np.sqrt(np.sum(((freqs[:, None] - c[None, :])**2) * m, axis=0) / (np.sum(m, axis=0) + 1e-10))

    def skewness(m, c, s):
        return np.sum(((freqs[:, None] - c[None, :])**3) * m, axis=0) / (np.sum(m, axis=0)*(s**3 + 1e-10))

    def kurtosis(m, c, s):
        return np.sum(((freqs[:, None] - c[None, :])**4) * m, axis=0) / (np.sum(m, axis=0)*(s**4 + 1e-10))

    def flatness(m):
        return np.exp(np.mean(np.log(m + 1e-10), axis=0)) / (np.mean(m, axis=0) + 1e-10)

    def crest(m):
        return np.max(m, axis=0) / (np.mean(m, axis=0) + 1e-10)

    def slope(m):
        k = np.arange(m.shape[0])[:, None]
        return np.sum(k * m, axis=0) / (np.sum(m, axis=0) + 1e-10)

    def decrease(m):
        k = np.arange(1, m.shape[0])[:, None]
        return np.sum((m[1:, :] - m[0:1, :]) / k, axis=0) / (np.sum(m[1:, :], axis=0) + 1e-10)

    def variation(m):
        return np.sqrt(np.mean(np.diff(m, axis=1)**2, axis=0))

    def zero_crossing(y):
        return librosa.feature.zero_crossing_rate(y + 1e-10)[0]

    def autocorr(y):
        result = np.correlate(y, y, mode='full')
        return result[result.size // 2:]

    def f0_and_harmonics(y, sr):
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        harmonic_energy = np.where(voiced_flag, f0, 0)
        return f0, harmonic_energy

    def inharmonicity(m):
        return np.var(np.diff(freqs))

    def harmonic_deviation(f0):
        return np.std(f0[np.isfinite(f0)]) if np.any(np.isfinite(f0)) else 0

    def odd_even_ratio(m):
        odds = m[1::2, :]
        evens = m[::2, :]
        return np.sum(odds, axis=0) / (np.sum(evens, axis=0) + 1e-10)

    # Compute descriptors
    energy = frame_energy(mag)
    spec_cent = centroid(mag)
    spec_spread = spread(mag, spec_cent)
    spec_skew = skewness(mag, spec_cent, spec_spread)
    spec_kurt = kurtosis(mag, spec_cent, spec_spread)
    spec_flat = flatness(mag)
    spec_crest = crest(mag)
    spec_slope = slope(mag)
    spec_decr = decrease(mag)
    spec_var = variation(mag)
    temp_cent = np.sum(time * energy) / (np.sum(energy) + 1e-10)

    # ADSR estimation using energy envelope
    norm_env = energy / (np.max(energy) + 1e-10)
    threshold = 0.1
    attack_idx = np.argmax(norm_env >= threshold)
    release_idx = len(norm_env) - np.argmax(norm_env[::-1] >= threshold) - 1

    att_time = time[attack_idx] if attack_idx < len(time) else 0.1
    rel_time = time[-1] - time[release_idx] if release_idx < len(time) else 0.2
    peak_idx = np.argmax(norm_env)
    dec_time = time[peak_idx] - att_time if peak_idx > attack_idx else 0.1
    sus_time = max(0.1, time[-1] - (att_time + dec_time + rel_time))

    zcr = zero_crossing(y)
    auto = autocorr(y)
    f0, harm_energy = f0_and_harmonics(y, sr)
    inharm = inharmonicity(mag)
    harm_dev = harmonic_deviation(f0)
    odd_even = odd_even_ratio(mag)

    noise_energy = energy - harm_energy
    noisiness = np.maximum(noise_energy, 0)

    desc = {
        'sr': sr,
        'duration': len(y)/sr,
        'Att': att_time,
        'Dec': dec_time,
        'Sus': sus_time,
        'Rel': rel_time,
        'TempCent': temp_cent,
        'Autocorrelation': auto,
        'ZCR': zcr,
        'FrameErg': energy,
        'AmpMod': np.std(energy),
        'FreqMod': np.mean(np.diff(energy)),
        'SpecCent': spec_cent,
        'SpecSpread': spec_spread,
        'SpecSkew': spec_skew,
        'SpecKurt': spec_kurt,
        'SpecDecr': spec_decr,
        'SpecFlat': spec_flat,
        'SpecCrest': spec_crest,
        'SpecSlope': spec_slope,
        'SpecVar': spec_var,
        'SpecShape': mag,
        'F0': f0,
        'HarmErg': harm_energy,
        'NoiseErg': noise_energy,
        'Noisiness': noisiness,
        'InHarm': inharm,
        'HarmDev': harm_dev,
        'OddEveRatio': odd_even,
    }
    return desc

# -------------------------------------------------
# === Additive Synthesis Engine ===
# -------------------------------------------------
def synthesize(desc, weights):
    sr = desc['sr']
    duration = desc['duration']
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = np.nan_to_num(desc['F0'])
    f0_interp = np.interp(t, np.linspace(0, duration, len(f0)), f0)

    # ADSR envelope
    att = max(weights['Att'] * desc['Att'], 0.01)
    dec = max(weights['Dec'] * desc['Dec'], 0.01)
    sus = max(weights['Sus'] * desc['Sus'], 0.01)
    rel = max(weights['Rel'] * desc['Rel'], 0.01)

    n_att = int(sr * att)
    n_dec = int(sr * dec)
    n_sus = int(sr * sus)
    n_rel = int(sr * rel)

    env = np.concatenate([
        np.linspace(0, 1, n_att, endpoint=False),
        np.linspace(1, 0.7, n_dec, endpoint=False),
        np.full(n_sus, 0.7),
        np.linspace(0.7, 0, n_rel, endpoint=False)
    ])
    env = np.pad(env, (0, max(0, len(t) - len(env))), mode='constant')[:len(t)]

    # Amplitude envelope modulated by energy-related descriptors
    amp_mod = weights['AmpMod'] * desc['AmpMod'] if 'AmpMod' in weights else 1.0
    freq_mod = weights['FreqMod'] * desc['FreqMod'] if 'FreqMod' in weights else 0.0

    # Noise band from ZCR and Noisiness
    noise = np.random.randn(len(t)) * weights['ZCR'] * np.mean(desc['ZCR']) * weights['Noisiness'] * np.mean(desc['Noisiness'])

    # Harmonic base signal
    signal = np.zeros_like(t)
    harmonics = int(weights.get('Harmonics', 10))
    for h in range(1, harmonics + 1):
        inharm_mod = (1 + weights['InHarm'] * desc['InHarm'])
        freq_shift = h * f0_interp * inharm_mod
        amp_weight = 1.0 / h
        if h % 2 == 1:
            amp_weight *= weights['OddEveRatio']
        else:
            amp_weight *= (1 - weights['OddEveRatio'])
        signal += amp_weight * np.sin(2 * np.pi * freq_shift * t + freq_mod)

    # Spectral shaping (basic filtering via spectral centroid & slope)
    spec_shape = weights['SpecCent'] * np.mean(desc['SpecCent']) + weights['SpecSlope'] * np.mean(desc['SpecSlope'])
    signal = signal * (1.0 - np.tanh(spec_shape * np.linspace(-1, 1, len(signal))))

    # Apply energy modulation and ADSR
    frame_energy = np.mean(desc['FrameErg'])
    signal = signal * env * weights['EnergyScale'] * frame_energy * amp_mod

    # Add noise
    signal += noise * weights.get('NoiseErg', 0.5)

    # Normalize
    signal /= np.max(np.abs(signal) + 1e-10)
    return signal, sr


# -------------------------------------------------
# === GUI Launcher ===
# -------------------------------------------------
def launch_gui():
    root = tk.Tk()
    root.title("Descriptor-Controlled Synth")
    root.geometry("1400x900")
    root.rowconfigure(2, weight=1)
    root.columnconfigure(0, weight=1)

    desc = {}
    sliders = {}

    time_varying_keys = [
        'FrameErg', 'SpecCent', 'SpecSpread', 'SpecSkew', 'SpecKurt', 'SpecDecr', 'SpecFlat',
        'SpecCrest', 'SpecSlope', 'SpecVar', 'SpecShape', 'F0', 'HarmErg', 'NoiseErg', 'Noisiness'
    ]

    static_keys = [
        'Att', 'Dec', 'Sus', 'Rel',                # ADSR
        'TempCent', 'Autocorrelation', 'ZCR',      # Temporal
        'AmpMod', 'FreqMod',                       # Energy
        'InHarm', 'HarmDev', 'OddEveRatio',        # Harmonics
        'NoiseErg', 'Noisiness',                   # Noise layer control
        'SpecCent', 'SpecSlope',                   # Spectral shaping
        'EnergyScale',                              # Global amplitude control
        'Harmonics' 
    ]


    def load_audio():
        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if not file_path:
            return
        nonlocal desc
        desc = extract_descriptors(file_path)
        for key, slider in sliders.items():
            if key in desc and isinstance(desc[key], (float, int, np.float32, np.float64)):
                val = float(desc[key])
                val = np.clip(val, 0.0, 1.0) if not np.isinf(val) else 0.5
                sliders[key].set(val)

        refresh_curve_thumbnails()

    def do_synthesize():
        weights = {key: slider.get() for key, slider in sliders.items() if key in desc or key == 'EnergyScale'}
        y, sr = synthesize(desc, weights)
        sf.write("synth_output.wav", y, sr)
        sd.play(y, samplerate=sr)

    def save_as():
        weights = {key: slider.get() for key, slider in sliders.items() if key in desc or key == 'EnergyScale'}
        y, sr = synthesize(desc, weights)
        save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if save_path:
            sf.write(save_path, y, sr)

    def refresh_curve_thumbnails():
        canvas_container.update_idletasks()
        panel_width = canvas_container.winfo_width()
        thumb_width = 320  # approx width per thumbnail including padding
        n_cols = max(1, panel_width // thumb_width)

        for widget in scroll_frame.winfo_children():
            widget.destroy()

        for idx, key in enumerate(time_varying_keys):
            if key not in desc or not isinstance(desc[key], np.ndarray):
                continue
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(desc[key])
            ax.set_title(key)
            ax.axis('off')
            canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
            fig.tight_layout()
            canvas.draw()
            plt.close(fig)
            widget = canvas.get_tk_widget()
            widget.grid(row=idx // n_cols, column=idx % n_cols, padx=5, pady=5)
            widget.bind("<Button-1>", lambda e, k=key: open_curve_editor(k))

    def open_curve_editor(key):
        top = tk.Toplevel(root)
        top.title(f"Edit Curve: {key}")
        fig, ax = plt.subplots(figsize=(8, 4))
        curve = np.array(desc[key], copy=True)
        line, = ax.plot(curve, marker='o')
        ax.set_title(f"Click and drag to edit {key}")

        def on_click(event):
            if event.inaxes != ax:
                return
            nearest_idx = int(round(event.xdata))
            if 0 <= nearest_idx < len(curve):
                curve[nearest_idx] = event.ydata
                line.set_ydata(curve)
                fig.canvas.draw()

        def apply_curve():
            desc[key] = np.clip(curve, 0, np.max(curve))
            top.destroy()
            refresh_curve_thumbnails()

        fig.canvas.mpl_connect("button_press_event", on_click)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack()
        ttk.Button(top, text="Apply", command=apply_curve).pack(pady=10)

    # UI Buttons
    ttk.Button(root, text="Load WAV", command=load_audio).pack(pady=5)
    ttk.Button(root, text="Synthesize & Play", command=do_synthesize).pack(pady=2)
    ttk.Button(root, text="Save As...", command=save_as).pack(pady=2)

    # Slider panel (13 sliders)
    slider_frame = ttk.LabelFrame(root, text="Static Descriptor Weights")
    slider_frame.pack(padx=10, pady=10, fill='x')

    num_columns = 2
    for idx, key in enumerate(static_keys):
        row = idx // num_columns
        col = idx % num_columns
        lbl = ttk.Label(slider_frame, text=key)
        lbl.grid(row=row, column=col*2, sticky='e', padx=5, pady=2)
        if key == 'Harmonics':
            slider = tk.Scale(slider_frame, from_=1, to=20, resolution=1, orient='horizontal', length=300)
            slider.set(10)
        elif key in ['SpecCent', 'SpecSlope', 'TempCent', 'FreqMod']:  # log-sensitive descriptors
            slider = tk.Scale(slider_frame, from_=0.001, to=1.0, resolution=0.001, orient='horizontal', length=300)
            slider.set(0.01)
        elif key == 'EnergyScale':
            slider = tk.Scale(slider_frame, from_=0, to=1.0, resolution=0.01, orient='horizontal', length=300)
            slider.set(1)
        else:
            slider = tk.Scale(slider_frame, from_=-10000, to=10000, resolution=1, orient='horizontal', length=300)
        slider.grid(row=row, column=col*2 + 1, sticky='w', padx=5, pady=2)
        sliders[key] = slider


    # Scrollable frame for 15 time-varying descriptor previews
    curve_panel = ttk.LabelFrame(root, text="Time-Varying Descriptor Curves (Click to Edit)")
    curve_panel.pack(fill='both', expand=True, padx=10, pady=10)

    canvas_container = tk.Canvas(curve_panel)
    scrollbar = tk.Scrollbar(curve_panel, orient="vertical", command=canvas_container.yview)
    scroll_frame = ttk.Frame(canvas_container)
    scroll_frame.bind("<Configure>", lambda e: canvas_container.configure(scrollregion=canvas_container.bbox("all")))

    canvas_container.create_window((0, 0), window=scroll_frame, anchor='nw')
    canvas_container.configure(yscrollcommand=scrollbar.set)
    canvas_container.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

if __name__ == '__main__':
    launch_gui()
