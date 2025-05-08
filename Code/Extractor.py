import numpy as np
import librosa
import librosa.display
from scipy.signal import correlate, savgol_filter
from scipy.stats import skew, kurtosis, linregress
from scipy.interpolate import interp1d
from scipy.fft import fft


def generate_adsr_curve(attack, decay, sustain_level, release, total_frames, sr, hop_length):
    env = np.zeros(total_frames)
    t = librosa.frames_to_time(np.arange(total_frames), sr=sr, hop_length=hop_length)
    total_dur = t[-1]

    a_end = attack
    d_end = a_end + decay
    s_end = total_dur - release

    for i, ti in enumerate(t):
        if ti < a_end:
            env[i] = (ti / attack) if attack > 0 else 1.0
        elif ti < d_end:
            env[i] = 1.0 - (1.0 - sustain_level) * ((ti - a_end) / decay)
        elif ti < s_end:
            env[i] = sustain_level
        else:
            env[i] = sustain_level * (1.0 - (ti - s_end) / release) if release > 0 else 0.0
    return env


def extract_descriptors_full(file_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    power = mag ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    time = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)

    frame_energy = np.sum(power, axis=0)
    norm_env = frame_energy / (np.max(frame_energy) + 1e-10)
    threshold = 0.4
    attack_idx = np.argmax(norm_env >= threshold)
    release_idx = len(norm_env) - np.argmax(norm_env[::-1] >= threshold) - 1
    peak_idx = np.argmax(norm_env)

    att_time = time[attack_idx] if attack_idx < len(time) else 0.1
    rel_time = time[-1] - time[release_idx] if release_idx < len(time) else 0.2
    dec_time = time[peak_idx] - att_time if peak_idx > attack_idx else 0.1
    sus_time = max(0.1, time[-1] - (att_time + dec_time + rel_time))
    sus_level = np.percentile(norm_env[peak_idx:release_idx], 75) if release_idx > peak_idx else norm_env[peak_idx]

    adsr_curve = generate_adsr_curve(att_time, dec_time, sus_level, rel_time, len(time), sr, hop_length)

    lat = np.log10(att_time + 1e-10)
    temp_cent = np.sum(time * frame_energy) / (np.sum(frame_energy) + 1e-10)
    amp_mod = np.std(frame_energy)
    freq_mod = np.mean(np.abs(np.diff(frame_energy)))

    env_fft = np.abs(fft(norm_env - np.mean(norm_env)))
    freq_bins = np.fft.fftfreq(len(norm_env), d=hop_length/sr)
    freq_mod_peak = freq_bins[np.argmax(env_fft[1:]) + 1]

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

    frame_size = n_fft
    autocorr = []
    for i in range(0, len(y) - frame_size, hop_length):
        frame = y[i:i + frame_size]
        ac = correlate(frame, frame, mode='full')[frame_size - 1:frame_size + 12]
        autocorr.append(ac / (ac[0] + 1e-10))
    autocorr = np.array(autocorr).T

    def spectral_moment(m, order):
        centroid = np.sum(freqs[:, None] * m, axis=0) / (np.sum(m, axis=0) + 1e-10)
        spread = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :])**2) * m, axis=0) / (np.sum(m, axis=0) + 1e-10))
        if order == 1:
            return centroid
        elif order == 2:
            return spread
        elif order == 3:
            norm = (freqs[:, None] - centroid[None, :]) / (spread[None, :] + 1e-10)
            return skew(norm * m, axis=0, bias=False)
        elif order == 4:
            norm = (freqs[:, None] - centroid[None, :]) / (spread[None, :] + 1e-10)
            return kurtosis(norm * m, axis=0, bias=False)

    spec_cent = spectral_moment(mag, 1)
    spec_spread = spectral_moment(mag, 2)
    spec_skew = spectral_moment(mag, 3)
    spec_kurt = spectral_moment(mag, 4)

    spec_flat = librosa.feature.spectral_flatness(S=mag)[0]
    spec_crest = np.max(mag, axis=0) / (np.mean(mag, axis=0) + 1e-10)
    spec_slope = np.array([linregress(freqs, mag[:, i])[0] for i in range(mag.shape[1])])
    spec_decr = np.sum((mag[1:, :] - mag[0:1, :]) / (np.arange(1, mag.shape[0])[:, None]), axis=0) / (np.sum(mag[1:, :], axis=0) + 1e-10)
    spec_var = np.sqrt(np.mean(np.diff(mag, axis=1)**2, axis=0))

    spec_flux_corr = []
    for i in range(1, mag.shape[1]):
        a = mag[:, i - 1]
        b = mag[:, i]
        corr = np.corrcoef(a, b)[0, 1]
        spec_flux_corr.append(1.0 - corr)
    spec_temp_var = np.array(spec_flux_corr)

    spec_shape_raw = np.mean(mag, axis=1)
    spec_shape_smooth = savgol_filter(spec_shape_raw, 9, 3)

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=n_fft, hop_length=hop_length)
    f0 = np.nan_to_num(f0)
    harm_energy = np.zeros_like(frame_energy)
    odd_energy = np.zeros_like(frame_energy)
    even_energy = np.zeros_like(frame_energy)
    inharm_vals = []
    harm_dev_vals = []

    interp_mag = [interp1d(freqs, mag[:, i], kind='linear', bounds_error=False, fill_value=0.0) for i in range(mag.shape[1])]

    for i, f in enumerate(f0):
        if f > 0:
            harmonics = np.array([(n + 1) * f for n in range(10)])
            amps = interp_mag[i](harmonics)
            harm_energy[i] = np.sum(amps)
            odd_energy[i] = np.sum(amps[::2])
            even_energy[i] = np.sum(amps[1::2])
            ideal = np.mean(amps)
            harm_dev_vals.append(np.std(amps - ideal))
            inharm = np.sum((harmonics - (np.arange(1, len(harmonics)+1) * f))**2 * amps**2) / (np.sum(amps**2) + 1e-10)
            inharm_vals.append((2 / f) * inharm)
        else:
            inharm_vals.append(0.0)
            harm_dev_vals.append(0.0)

    inharm = np.array(inharm_vals)
    harm_dev = np.array(harm_dev_vals)
    noise_energy = frame_energy - harm_energy
    noisiness = np.clip(noise_energy / (frame_energy + 1e-10), 0, 1)
    odd_even = np.clip(odd_energy / (even_energy + 1e-10), 0, 10)

    descriptors = {
        'sr': sr,
        'duration': len(y)/sr,
        'ADSR_Curve': adsr_curve,
        'Att': att_time,
        'Dec': dec_time,
        'SusTime': sus_time,
        'SusLevel': sus_level,
        'Rel': rel_time,
        'TempCent': temp_cent,
        'Autocorrelation': autocorr,
        'ZCR': zcr,
        'FrameErg': frame_energy,
        'AmpMod': amp_mod,
        'FreqMod': freq_mod_peak,
        'SpecCent': spec_cent,
        'SpecSpread': spec_spread,
        'SpecSkew': spec_skew,
        'SpecKurt': spec_kurt,
        'SpecDecr': spec_decr,
        'SpecFlat': spec_flat,
        'SpecCrest': spec_crest,
        'SpecSlope': spec_slope,
        'SpecVar': spec_var,
        'SpectroTempVar': spec_temp_var,
        'SpecShape': spec_shape_smooth,
        'F0': f0,
        'HarmErg': harm_energy,
        'NoiseErg': noise_energy,
        'Noisiness': noisiness,
        'InHarm': inharm,
        'HarmDev': harm_dev,
        'OddEveRatio': odd_even,
    }

    return descriptors