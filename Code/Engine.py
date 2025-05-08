# descriptor_synthesis_engine.py

import numpy as np
from scipy.signal import butter, lfilter, stft, istft, get_window

# === Modular Descriptor Control (Curves) ===
def descriptor_curve(length, value):
    if isinstance(value, (float, int)):
        return np.ones(length) * value
    value = np.asarray(value)
    if value.ndim == 1:
        return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(value)), value)
    elif value.ndim == 2:
        # Interpolate each row independently (assume shape [lags, frames])
        interpolated = np.array([
            np.interp(np.linspace(0, 1, length), np.linspace(0, 1, value.shape[1]), row)
            for row in value
        ])
        return interpolated
    else:
        raise ValueError("Unsupported descriptor array shape for interpolation")

# === Temporal Blending Utility ===
def temporal_blend(prev, curr, alpha):
    return (1 - alpha) * prev + alpha * curr

# === ADSR Envelope Generator with Temporal Centroid Bias ===
def adsr_curve(attack, decay, sustain_level, release, total_len, sr, centroid=0.5):
    attack *= (1 - centroid)
    release *= centroid
    a_len = int(attack * sr)
    d_len = int(decay * sr)
    r_len = int(release * sr)
    s_len = max(total_len - (a_len + d_len + r_len), 1)
    a = np.linspace(0, 1, a_len)
    d = np.linspace(1, sustain_level, d_len)
    s = np.ones(s_len) * sustain_level
    r = np.linspace(sustain_level, 0, r_len)
    return np.concatenate([a, d, s, r])[:total_len]

# === Harmonic Synth ===
def harmonic_bank(f0_curve, inharm_curve, odd_even_curve, amp_decay_curve, harm_dev_curve, autocorr_curve, frame_count, block_size, sr):
    output = np.zeros((frame_count, block_size))
    partials = 20
    for i in range(frame_count):
        t_frame = np.arange(block_size) / sr
        f0 = f0_curve[i]
        inharm = inharm_curve[i]
        odd_even = odd_even_curve[i]
        amp_decay = amp_decay_curve[i]
        harm_dev = harm_dev_curve[i]
        # autocorr = autocorr_curve[i]
        
        autocorr_slice = autocorr_curve[:, i]  # shape (lags,)
        autocorr_interp = np.interp(
            np.linspace(0, 1, block_size),
            np.linspace(0, 1, autocorr_slice.shape[0]),
            autocorr_slice
        )

        frame = np.zeros(block_size)
        for h in range(1, partials + 1):
            freq = f0 * h * np.sqrt(1 + inharm * h**2)
            if freq >= sr / 2:
                continue
            phase_mod = autocorr_interp * np.sin(2 * np.pi * freq * t_frame)
            amp = (1.0 / (h ** amp_decay)) * (1 + np.random.uniform(-harm_dev, harm_dev))
            amp *= odd_even if h % 2 else (1 - odd_even)
            frame += amp * np.sin(2 * np.pi * freq * t_frame + phase_mod)
        output[i] = frame
    return output

# === Noise Synth ===
def noise_bandpass(zcr_curve, frame_count, sr, block_size):
    nyq = sr / 2
    output = np.zeros((frame_count, block_size))
    for i in range(frame_count):
        bw = zcr_curve[i] * 1000
        center_freq = 1000 + zcr_curve[i] * 4000
        low_mod = center_freq - bw / 2
        high_mod = center_freq + bw / 2

        # Ensure valid bounds
        low_mod = max(20, min(low_mod, nyq - 2))
        high_mod = max(low_mod + 1, min(high_mod, nyq - 1))

        try:
            b, a = butter(2, [low_mod / nyq, high_mod / nyq], btype='band')
            noise = np.random.randn(block_size)
            filtered = lfilter(b, a, noise)
            output[i] = filtered
        except ValueError as e:
            print(f"Warning: invalid bandpass params (low={low_mod}, high={high_mod}), skipping frame {i}")
            output[i] = np.zeros(block_size)
    return output


# === Spectral Envelope Shaping ===
def apply_spectral_envelope(mag, spec_cent, spec_spread, spec_skew, spec_kurt, spec_slope, spec_decrease, spec_flat, spec_crest, spec_var, spec_temp_var, spec_shape, spec_flux_curve, spec_shape_morph, sr):
    f_bins = np.linspace(0, sr / 2, mag.shape[1])
    env = np.zeros_like(mag)
    prev_values = {
        'center': spec_cent[0],
        'spread': spec_spread[0],
        'skew': spec_skew[0],
        'kurt': spec_kurt[0],
        'slope': spec_slope[0],
        'decrease': spec_decrease[0]
    }
    for i in range(mag.shape[0]):
        alpha = spec_temp_var[i]
        center = temporal_blend(prev_values['center'], spec_cent[i], alpha)
        spread = temporal_blend(prev_values['spread'], spec_spread[i], alpha)
        skew = temporal_blend(prev_values['skew'], spec_skew[i], alpha)
        kurt = temporal_blend(prev_values['kurt'], spec_kurt[i], alpha)
        slope = temporal_blend(prev_values['slope'], spec_slope[i], alpha)
        decrease = temporal_blend(prev_values['decrease'], spec_decrease[i], alpha)

        prev_values.update({'center': center, 'spread': spread, 'skew': skew, 'kurt': kurt, 'slope': slope, 'decrease': decrease})

        flat = spec_flat[i]
        crest = spec_crest[i]
        shape = spec_shape[i]
        dist = (f_bins - center) / (spread + 1e-6)
        skewness = dist ** 3 * skew
        peaked = np.exp(-0.5 * (dist ** 2)) * (1 + skewness)
        peaked /= np.max(peaked)
        peaked = peaked ** (1.0 / (kurt + 1e-6))
        tilt = np.interp(f_bins, [0, sr / 2], [1 + slope, 1 - decrease])
        env_i = peaked * tilt
        env_i = env_i * (1 - flat) + flat * np.random.uniform(0.5, 1.0, size=env_i.shape)
        env_i = np.clip(env_i * crest * shape, 0, 1)

        jitter = np.random.normal(0, spec_var[i] * 0.1, size=env_i.shape)
        env_i = np.clip(env_i + jitter, 0, 1)

        if i > 0:
            flux = np.abs(env_i - env[i - 1])
            env_i += flux * spec_flux_curve[i]

        env_i = temporal_blend(np.ones_like(env_i), env_i, spec_shape_morph[i])

        env[i] = env_i
    return mag * env

# === Synthesis Core ===
def synthesize(param_dict, descriptor_data=None):
    
    def get_strength(param_dict, label, default=1.0):
        return param_dict.get(f"{label} Strength", default)

    sr = int(param_dict.get("Sample Rate", 44100))
    block_size = int(param_dict.get("Block Size", 1024))
    hop_size = int(param_dict.get("Hop Size", 512))
    
    descriptor_len = len(descriptor_data["FrameErg"])
    duration = (descriptor_len * hop_size) / sr

    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    frame_count = int(np.ceil((n_samples - block_size) / hop_size)) + 1

    f0_curve = descriptor_curve(frame_count, descriptor_data.get("Fundamental Frequency", 220)) * get_strength(
        param_dict, "Fundamental Frequency")
    inharm_curve = descriptor_curve(frame_count, descriptor_data.get("Inharmonicity", 0.1)) * get_strength(param_dict,
                                                                                                           "Inharmonicity")
    odd_even_curve = descriptor_curve(frame_count,
                                      descriptor_data.get("Odd to Even Harmonic Ratio", 0.6)) * get_strength(param_dict,
                                                                                                             "Odd to Even Harmonic Ratio")
    amp_decay_curve = descriptor_curve(frame_count, param_dict.get("Amp Decay", 1.2))
    harm_dev_curve = descriptor_curve(frame_count, param_dict.get("Harmonic Spectral Deviation", 0.0))
    autocorr_curve = descriptor_curve(frame_count, descriptor_data.get("Autocorrelation", 0.0)) * get_strength(
        param_dict, "Autocorrelation")

    harm_energy_curve = descriptor_curve(frame_count, descriptor_data.get("Harmonic Energy", 0.7)) * get_strength(
        param_dict, "Harmonic Energy")
    noise_energy_curve = descriptor_curve(frame_count, descriptor_data.get("Noise Energy", 0.3)) * get_strength(
        param_dict, "Noise Energy")
    frame_energy_curve = descriptor_curve(frame_count, descriptor_data.get("Frame Energy", 1.0)) * get_strength(
        param_dict, "Frame Energy")
    noisiness_modifier = get_strength(param_dict, "Noisiness")  # Optional curve
    noisiness_curve = (frame_energy_curve - harm_energy_curve) * noisiness_modifier

    zcr_curve = descriptor_curve(frame_count, descriptor_data.get("Zero Crossing Rate", 0.0)) * get_strength(param_dict,
                                                                                                             "Zero Crossing Rate")
    spec_cent = descriptor_curve(frame_count, descriptor_data.get("Spectral Centroid", 1000)) * get_strength(param_dict,
                                                                                                             "Spectral Centroid")
    spec_spread = descriptor_curve(frame_count, descriptor_data.get("Spectral Spread", 1000)) * get_strength(param_dict,
                                                                                                             "Spectral Spread")
    spec_skew = descriptor_curve(frame_count, descriptor_data.get("Spectral Skewness", 0.0)) * get_strength(param_dict,
                                                                                                            "Spectral Skewness")
    spec_kurt = descriptor_curve(frame_count, descriptor_data.get("Spectral Kurtosis", 3.0)) * get_strength(param_dict,
                                                                                                            "Spectral Kurtosis")
    spec_slope = descriptor_curve(frame_count, descriptor_data.get("Spectral Slope", 0.0)) * get_strength(param_dict,
                                                                                                          "Spectral Slope")
    spec_decrease = descriptor_curve(frame_count, descriptor_data.get("Spectral Decrease", 0.5)) * get_strength(
        param_dict, "Spectral Decrease")
    spec_flat = descriptor_curve(frame_count, descriptor_data.get("Spectral Flatness", 0.5)) * get_strength(param_dict,
                                                                                                            "Spectral Flatness")
    spec_crest = descriptor_curve(frame_count, descriptor_data.get("Spectral Crest", 10.0)) * get_strength(param_dict,
                                                                                                           "Spectral Crest")
    spec_var = descriptor_curve(frame_count, descriptor_data.get("Spectral Variation", 0.0)) * get_strength(param_dict,
                                                                                                            "Spectral Variation")
    spec_temp_var = descriptor_curve(frame_count,
                                     descriptor_data.get("Spectro-temporal Variation", 0.0)) * get_strength(param_dict,
                                                                                                            "Spectro-temporal Variation")
    spec_shape = descriptor_curve(frame_count, descriptor_data.get("Spectral Shape", 1.0)) * get_strength(param_dict,
                                                                                                          "Spectral Shape")
    spec_flux_curve = descriptor_curve(frame_count, descriptor_data.get("Spectral Flux", 0.0)) * get_strength(
        param_dict, "Spectral Flux")
    spec_shape_morph = descriptor_curve(frame_count, descriptor_data.get("Spectral Shape Morph", 1.0)) * get_strength(
        param_dict, "Spectral Shape Morph")

    temp_centroid_curve = descriptor_curve(frame_count, descriptor_data.get("Temporal Centroid", 0.5)) * get_strength(
        param_dict, "Temporal Centroid")
    tremolo_depth_curve = descriptor_curve(frame_count, param_dict.get("Tremolo Depth", 0.0))
    tremolo_freq_curve = descriptor_curve(frame_count,
                                          descriptor_data.get("Frequency of Energy Modulation", 5.0)) * get_strength(
        param_dict, "Frequency of Energy Modulation")

    effective_duration_curve = descriptor_curve(frame_count,
                                                descriptor_data.get("Effective Duration", 1.0)) * get_strength(
        param_dict, "Effective Duration")

    # adsr = adsr_curve(0.01, 0.2, 0.7, 0.3, n_samples, sr, centroid=temp_centroid_curve.mean())
    # adsr = adsr[:int(effective_duration_curve.mean() * sr)]

    harm_frames = harmonic_bank(f0_curve, inharm_curve, odd_even_curve, amp_decay_curve, harm_dev_curve, autocorr_curve, frame_count, block_size, sr)
    noise_frames = noise_bandpass(zcr_curve, frame_count, sr, block_size)

    mixed = harm_frames * harm_energy_curve[:, None] + noise_frames * noise_energy_curve[:, None] * noisiness_curve[:,
                                                                                                    None]

    window = get_window("hann", block_size)
    stft_frames = np.array([np.fft.rfft(frame * window) for frame in mixed])
    mag = np.abs(stft_frames)
    phase = np.angle(stft_frames)
    shaped_mag = apply_spectral_envelope(mag, spec_cent, spec_spread, spec_skew, spec_kurt, spec_slope, spec_decrease, spec_flat, spec_crest, spec_var, spec_temp_var, spec_shape, spec_flux_curve, spec_shape_morph, sr)
    shaped_stft = shaped_mag * np.exp(1j * phase)

    output = np.zeros((frame_count - 1) * hop_size + block_size)
    for i in range(frame_count):
        frame_time = np.fft.irfft(shaped_stft[i])
        start = i * hop_size
        output[start:start + block_size] += frame_time * window * frame_energy_curve[i]

    trem_t = np.linspace(0, duration, len(output))
    tremolo = 1 + tremolo_depth_curve[0] * np.sin(2 * np.pi * tremolo_freq_curve[0] * trem_t)
    output *= tremolo[:len(output)]

    # output *= adsr[:len(output)]
    adsr_len = int(effective_duration_curve.mean() * sr)
    raw_adsr = adsr_curve(0.01, 0.2, 0.7, 0.3, adsr_len, sr, centroid=temp_centroid_curve.mean())

    # Interpolate to match output shape
    adsr = np.interp(
        np.linspace(0, 1, len(output)),
        np.linspace(0, 1, len(raw_adsr)),
        raw_adsr
    )
    
    output *= adsr
    output /= np.max(np.abs(output)) + 1e-9

    return output