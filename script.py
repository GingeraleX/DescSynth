import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

# -------------------------------
# USER PARAMETERS - DEFAULTS
# -------------------------------
AUDIO_FILE_DEFAULT = 'input.wav'  # Path to your audio file
N_FFT_DEFAULT = 2048  # FFT size
HOP_LENGTH_DEFAULT = 512  # Hop size
WINDOW_TYPE_DEFAULT = 'hann'  # Window type
SYNTHESIS_MODE_DEFAULT = 'model_based' # Default synthesis mode
# Options: 'direct', 'griffin_lim', 'model_based'

# Number of iterations for Griffin-Lim
GRIFFIN_ITER_DEFAULT = 32

# Thresholds/Parameters for partial-based analysis
PEAK_THRESHOLD_RATIO_DEFAULT = 0.3  # Ratio of the max amplitude in a frame for partial detection
MAX_PARTIALS_DEFAULT = 20  # Maximum partials to consider
F0_RANGE_DEFAULT = (50.0, 2000.0)  # Frequency range for fundamental frequency search

# For "Energy Modulation" (tremolo), we look in the 1-10 Hz range
TREM_MIN_FREQ_DEFAULT = 1.0
TREM_MAX_FREQ_DEFAULT = 10.0


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_audio(filepath):
    """Load audio as mono signal."""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr


def stft_analysis(y, n_fft, hop, window):
    """Compute complex STFT."""
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, window=window)


def istft_synthesis(D, hop, window):
    """Inverse STFT for direct reconstruction."""
    return librosa.istft(D, hop_length=hop, window=window)


# -------------------------------
# GLOBAL DESCRIPTORS
# -------------------------------
def estimate_attack_time(energy, sr, hop, method='simple_threshold'):
    """
    Estimate attack time.
    'simple_threshold': pick first frame that exceeds 10% of max energy.
    A more advanced 'weakest-effort' method could be implemented here.
    """
    threshold = 0.1 * np.max(energy)
    frames_above = np.where(energy >= threshold)[0]
    if len(frames_above) > 0:
        first_frame = frames_above[0]
        return first_frame * hop / sr
    return 0.0


def estimate_release_time(energy, sr, hop):
    """
    Estimate release time as the last frame above 10% of max energy.
    """
    threshold = 0.1 * np.max(energy)
    frames_above = np.where(energy >= threshold)[0]
    if len(frames_above) > 0:
        last_frame = frames_above[-1]
        return last_frame * hop / sr
    return 0.0


def estimate_decay_time(energy, sr, hop):
    """
    Estimate 'decay time' as time from peak to some fraction (e.g. 50%)
    of that peak. Simplified approach for demonstration.
    """
    peak_idx = np.argmax(energy)
    peak_val = energy[peak_idx]
    half_val = 0.5 * peak_val
    frames_below = np.where(energy[peak_idx:] < half_val)[0]
    if len(frames_below) > 0:
        return (frames_below[0] + peak_idx) * hop / sr
    return 0.0


def log_attack_time(attack_time):
    """Log Attack Time in base 10."""
    if attack_time <= 0:
        return 0.0
    return np.log10(attack_time)


def attack_slope(energy, sr, hop, attack_time):
    """
    Approximate slope from start to attack_time.
    For demonstration, we do a linear regression approach on that segment.
    """
    if attack_time <= 0:
        return 0.0
    attack_frames = int(attack_time * sr / hop)
    segment = energy[:attack_frames + 1]
    if len(segment) < 2:
        return 0.0
    x = np.arange(len(segment))
    slope = np.polyfit(x, segment, 1)[0]
    return slope


def decrease_slope(energy, sr, hop):
    """
    Approximate the slope after the peak until the end (exponential-like).
    We'll do a linear fit in log domain.
    """
    peak_idx = np.argmax(energy)
    tail = energy[peak_idx:]
    tail_nonzero = tail[tail > 0]
    if len(tail_nonzero) < 2:
        return 0.0
    # log of energy
    log_tail = np.log(tail_nonzero)
    x = np.arange(len(log_tail))
    slope = np.polyfit(x, log_tail, 1)[0]
    return slope


def effective_duration(energy, sr, hop):
    """
    Effective duration = time where energy is above a threshold (e.g. 40% of max).
    """
    threshold = 0.4 * np.max(energy)
    frames_above = np.where(energy >= threshold)[0]
    if len(frames_above) == 0:
        return 0.0
    dur_frames = frames_above[-1] - frames_above[0]
    return dur_frames * hop / sr


def energy_modulation_tremolo(energy, sr, hop):
    """
    Estimate amplitude and frequency of amplitude modulation (tremolo)
    in the 1-10 Hz range. We'll do a naive DFT on the energy envelope.
    """
    # Convert frame-based energy to a signal in time:
    # Each frame is hop samples, we just upsample or treat as a discrete signal at sr_frame
    sr_frame = sr / hop  # frames per second
    # We'll do a naive real FFT on energy
    spectrum = np.fft.fft(energy)
    freqs = np.fft.fftfreq(len(energy), d=1.0 / sr_frame)
    # Keep only positive freqs
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    spec_pos = np.abs(spectrum[pos_mask])

    # Find the peak in 1-10 Hz
    band_mask = (freqs_pos >= TREM_MIN_FREQ_DEFAULT) & (freqs_pos <= TREM_MAX_FREQ_DEFAULT)
    if not np.any(band_mask):
        return (0.0, 0.0)
    band_spec = spec_pos[band_mask]
    band_freqs = freqs_pos[band_mask]
    peak_idx = np.argmax(band_spec)
    amp = band_spec[peak_idx]
    f_mod = band_freqs[peak_idx]
    return (amp, f_mod)


# -------------------------------
# TIME-VARYING SPECTRAL DESCRIPTORS
# -------------------------------
def spectral_variation(magnitude):
    """
    Spectral variation (flux) frame to frame:
    1 - correlation(previous_frame, current_frame).
    Returns an array of length (num_frames-1).
    """
    num_frames = magnitude.shape[1]
    var = np.zeros(num_frames - 1)
    for i in range(1, num_frames):
        prev = magnitude[:, i - 1]
        curr = magnitude[:, i]
        num = np.sum(prev * curr)
        den = np.sqrt(np.sum(prev ** 2) * np.sum(curr ** 2))
        if den > 0:
            corr = num / den
        else:
            corr = 0
        var[i - 1] = 1 - corr
    return var


def spectral_crest(magnitude):
    """
    Spectral crest measure (per frame):
    max(mag) / mean(mag)
    """
    # shape: (freq_bins, frames)
    max_val = np.max(magnitude, axis=0)
    mean_val = np.mean(magnitude, axis=0) + 1e-10  # avoid div0
    crest = max_val / mean_val
    return crest


# -------------------------------
# PARTIAL-BASED DESCRIPTORS
# -------------------------------
def partial_based_analysis(D, sr, max_partials=20, f0_range=(50, 2000)):
    """
    Naive partial-based analysis:
    - Estimate fundamental frequency (f0) per frame using e.g. librosa.pyin or a simpler method.
    - For each frame, pick up to 'max_partials' peaks in magnitude spectrum.
    - Estimate partial frequencies, check their closeness to integer multiples of f0 (-> inharmonicity).
    - Compute odd/even ratio, harmonic spectral deviation, noisiness, etc.
    """
    y_approx = librosa.istft(D)  # we do a rough time-domain signal for F0
    # We can attempt f0 detection using librosa.pyin (if installed).
    # For demonstration, let's do a simpler approach with librosa.yin:
    f0_series = librosa.yin(y_approx, sr=sr, frame_length=2048, hop_length=512, fmin=f0_range[0], fmax=f0_range[1])

    magnitude = np.abs(D)
    n_bins, n_frames = magnitude.shape
    freqs = np.linspace(0, sr / 2, 1 + n_bins // 2)

    # Arrays to store partial-based descriptors
    inharmonicity_arr = np.zeros(n_frames)
    odd_even_ratio_arr = np.zeros(n_frames)
    noiseness_arr = np.zeros(n_frames)
    harmonic_dev_arr = np.zeros(n_frames)

    for i in range(n_frames):
        # fundamental frequency in this frame
        f0_est = f0_series[i] if not np.isnan(f0_series[i]) else 0.0
        # We'll do partial detection by picking peaks in half-spectrum:
        half_spectrum = magnitude[:n_bins // 2 + 1, i]
        # find peaks
        peak_thresh = PEAK_THRESHOLD_RATIO_DEFAULT * np.max(half_spectrum)
        peaks, _ = find_peaks(half_spectrum, height=peak_thresh)

        if len(peaks) == 0 or f0_est <= 0:
            # if no peaks or no f0, skip
            inharmonicity_arr[i] = 0
            odd_even_ratio_arr[i] = 1
            noiseness_arr[i] = 1
            harmonic_dev_arr[i] = 0
            continue

        # Sort peaks by amplitude descending
        amps = half_spectrum[peaks]
        sorted_idx = np.argsort(amps)[::-1]
        top_peaks = peaks[sorted_idx][:max_partials]
        # approximate partial freq
        partial_freqs = freqs[top_peaks]
        partial_amps = half_spectrum[top_peaks]

        # compute total harmonic energy
        harmonic_energy = np.sum(partial_amps ** 2)
        total_energy = np.sum(half_spectrum ** 2)
        if total_energy == 0:
            noiseness = 1
        else:
            noiseness = 1 - (harmonic_energy / total_energy)

        # inharmonicity: sum of deviation^2 from integer multiples
        # B = sum_{h=1..H} [ (f_h - h*f0)^2 / f_h^2 ], weighted by amplitude
        # We'll do a simpler version: sum_{h=1..H} (|f_h - h*f0|)
        # Identify which partial is "harmonic number h"
        # This is naive: we assume partial_freqs are sorted by freq.
        partials_sorted = sorted(zip(partial_freqs, partial_amps), key=lambda x: x[0])
        inharm = 0
        sum_amp = 0
        h = 1
        for (pfreq, pamp) in partials_sorted:
            # expected freq = h*f0_est
            dev = abs(pfreq - h * f0_est)
            inharm += dev * pamp
            sum_amp += pamp
            h += 1
        if sum_amp > 0:
            inharmonicity = inharm / sum_amp
        else:
            inharmonicity = 0

        # odd-even ratio: sum of amps of odd partials / sum of amps of even partials
        # again naive, using the index in partials_sorted as the partial order
        # if partial i is "h" -> odd/even depends on h
        odd_amp = 0
        even_amp = 0
        h2 = 1
        for (pfreq, pamp) in partials_sorted:
            if h2 % 2 == 1:
                odd_amp += pamp
            else:
                even_amp += pamp
            h2 += 1
        if even_amp == 0:
            odd_even_ratio = 9999.0
        else:
            odd_even_ratio = odd_amp / even_amp

        # harmonic spectral deviation: difference from smoothed envelope
        # We'll do a naive approach: if we had partials sorted by freq,
        # we can approximate a local average for each partial.
        # Here, let's skip a detailed approach and define 0.
        # In a real system, you'd compute a spectral envelope and measure the difference.
        harmonic_dev = 0.0

        inharmonicity_arr[i] = inharmonicity
        odd_even_ratio_arr[i] = odd_even_ratio
        noiseness_arr[i] = noiseness
        harmonic_dev_arr[i] = harmonic_dev

    partial_descriptors = {
        'f0_series': f0_series,
        'inharmonicity': inharmonicity_arr,
        'odd_even_ratio': odd_even_ratio_arr,
        'noiseness': noiseness_arr,
        'harmonic_dev': harmonic_dev_arr
    }
    return partial_descriptors


# -------------------------------
# DESCRIPTOR EXTRACTION (FULL)
# -------------------------------
def extract_all_descriptors(D, sr, hop, n_fft):
    """
    Extract the selected descriptors:
      GLOBAL: Attack, Decay, Release, LogAttackTime, AttackSlope, DecreaseSlope,
              EffectiveDuration, EnergyModulation
      TIME-VARYING: Spectral Centroid, Spread, Skewness, Kurtosis, Rolloff,
                    Flatness, Crest, Variation (Flux)
      PARTIAL-BASED: Noiseness, FundamentalFrequency, Inharmonicity,
                     OddEvenRatio, HarmonicSpectralDeviation
    """
    # Magnitude
    magnitude = np.abs(D)
    # Phase (if needed for direct)
    phase = np.angle(D)

    # Frame-based energy
    frame_energy = np.sum(magnitude ** 2, axis=0)

    # ========== GLOBAL DESCRIPTORS ==========
    # Attack
    attack_time = estimate_attack_time(frame_energy, sr, hop)
    # Decay
    decay_time = estimate_decay_time(frame_energy, sr, hop)
    # Release
    release_time = estimate_release_time(frame_energy, sr, hop)
    # Log Attack Time
    lat = log_attack_time(attack_time)
    # Attack Slope
    att_slope = attack_slope(frame_energy, sr, hop, attack_time)
    # Decrease Slope
    dec_slope = decrease_slope(frame_energy, sr, hop)
    # Effective Duration
    eff_dur = effective_duration(frame_energy, sr, hop)
    # Energy Modulation (Tremolo)
    trem_amp, trem_freq = energy_modulation_tremolo(frame_energy, sr, hop)

    # ========== TIME-VARYING SPECTRAL DESCRIPTORS ==========
    # Centroid
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    # Spread (bandwidth in librosa, which is standard deviation)
    spread = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, hop_length=hop, n_fft=n_fft, roll_percent=0.85)[0]
    # Flatness
    flatness = librosa.feature.spectral_flatness(S=magnitude, hop_length=hop, n_fft=n_fft)[0]
    # Crest
    crest = spectral_crest(magnitude)
    # Variation (flux)
    var_flux = spectral_variation(magnitude)

    # Skewness and Kurtosis (per frame)
    num_frames = magnitude.shape[1]
    skewness_arr = np.zeros(num_frames)
    kurtosis_arr = np.zeros(num_frames)
    # We'll use half-spectrum approach
    half_bins = n_fft // 2 + 1
    for i in range(num_frames):
        spec = magnitude[:half_bins, i]
        if np.sum(spec) > 0:
            spec_norm = spec / np.sum(spec)
            skewness_arr[i] = skew(spec_norm)
            kurtosis_arr[i] = kurtosis(spec_norm)  # excess kurtosis
        else:
            skewness_arr[i] = 0
            kurtosis_arr[i] = 0

    # ========== PARTIAL-BASED DESCRIPTORS ==========
    partial_desc = partial_based_analysis(D, sr, max_partials=MAX_PARTIALS_DEFAULT, f0_range=F0_RANGE_DEFAULT)

    # Pack everything
    descriptors = {
        # Global
        'attack_time': attack_time,
        'decay_time': decay_time,
        'release_time': release_time,
        'log_attack_time': lat,
        'attack_slope': att_slope,
        'decrease_slope': dec_slope,
        'effective_duration': eff_dur,
        'tremolo_amplitude': trem_amp,
        'tremolo_frequency': trem_freq,

        # Time-varying
        'spectral_centroid': centroid,
        'spectral_spread': spread,
        'spectral_rolloff': rolloff,
        'spectral_flatness': flatness,
        'spectral_crest': crest,
        'spectral_variation': var_flux,  # length = frames-1
        'spectral_skewness': skewness_arr,
        'spectral_kurtosis': kurtosis_arr,

        # Partial-based
        'f0_series': partial_desc['f0_series'],
        'inharmonicity': partial_desc['inharmonicity'],
        'odd_even_ratio': partial_desc['odd_even_ratio'],
        'noiseness': partial_desc['noiseness'],
        'harmonic_dev': partial_desc['harmonic_dev']
    }
    return descriptors, phase


# -------------------------------
# SYNTHESIS METHODS
# -------------------------------
def synthesis_direct(D):
    """Direct inversion with the original STFT."""
    return istft_synthesis(D, HOP_LENGTH_DEFAULT, WINDOW_TYPE_DEFAULT)


def synthesis_griffin_lim(magnitude, sr, n_fft, hop_length, window_type, griffin_iter):
    """Griffin-Lim iterative reconstruction from magnitude only."""
    y_rec = librosa.griffinlim(magnitude,
                               n_iter=griffin_iter,
                               hop_length=hop_length,
                               win_length=n_fft, # Added win_length to match n_fft
                               window=window_type)
    return y_rec


def synthesis_model_based(magnitude, sr, n_fft, hop_length, window_type):
    """
    Model-based sinusoidal approach:
    For each frame, detect peaks and create a partial-based reconstruction
    with zero phase. Overlap-add the frames. Simplified example.
    """
    n_bins, n_frames = magnitude.shape
    freq_vec = np.linspace(0, sr / 2, n_bins)

    # Output length
    y_len = (n_frames - 1) * hop_length + n_fft
    y_out = np.zeros(y_len)

    # Synthesis window
    win = librosa.filters.get_window(window_type, n_fft, fftbins=True)

    for i in range(n_frames):
        frame_mag = magnitude[:, i]
        # Peak picking
        peak_thresh = PEAK_THRESHOLD_RATIO_DEFAULT * np.max(frame_mag)
        peaks, _ = find_peaks(frame_mag, height=peak_thresh)

        # Build a complex half-spectrum
        half_spec = np.zeros(n_bins, dtype=np.complex128)
        for p in peaks:
            amp = frame_mag[p]
            half_spec[p] = amp * np.exp(1j * 0.0)  # zero phase for simplicity

        # Mirror to full spectrum
        full_spec = np.concatenate([half_spec, np.conj(half_spec[1:-1][::-1])])

        # iFFT
        frame_time = np.real(np.fft.ifft(full_spec))
        # Window
        frame_time *= win
        # Overlap-add
        start = i * hop_length
        end = start + n_fft
        y_out[start:end] += frame_time

    return y_out


# -------------------------------
# MAIN
# -------------------------------
def main():
    # --- PARAMETERS ---
    audio_file = AUDIO_FILE_DEFAULT
    n_fft = N_FFT_DEFAULT
    hop_length = HOP_LENGTH_DEFAULT
    window_type = WINDOW_TYPE_DEFAULT
    synthesis_mode_str = SYNTHESIS_MODE_DEFAULT
    griffin_iter = GRIFFIN_ITER_DEFAULT
    peak_threshold_ratio = PEAK_THRESHOLD_RATIO_DEFAULT
    max_partials = MAX_PARTIALS_DEFAULT
    f0_range = F0_RANGE_DEFAULT
    trem_min_freq = TREM_MIN_FREQ_DEFAULT
    trem_max_freq = TREM_MAX_FREQ_DEFAULT

    print("Loading audio...")
    y, sr = load_audio(audio_file)
    print(f"Audio loaded: {audio_file}, sr={sr}, duration={len(y) / sr:.2f}s")

    # --- SYNTHESIS MODE SELECTION ---
    print("\nAvailable synthesis modes:")
    print("1: direct")
    print("2: griffin_lim")
    print("3: model_based")
    while True:
        mode_choice = input("Choose synthesis mode (1, 2, or 3) or press Enter for default (model_based): ").strip()
        if mode_choice == '1':
            synthesis_mode_str = 'direct'
            break
        elif mode_choice == '2':
            synthesis_mode_str = 'griffin_lim'
            break
        elif mode_choice == '3':
            synthesis_mode_str = 'model_based'
            break
        elif mode_choice == '':
            synthesis_mode_str = SYNTHESIS_MODE_DEFAULT
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or Enter.")

    # --- PARAMETER MODIFICATION ---
    modify_params = input("Do you want to modify synthesis parameters? (y/n): ").strip().lower()
    if modify_params == 'y':
        print("\n--- Current Parameters ---")
        print(f"1: N_FFT = {n_fft}")
        print(f"2: HOP_LENGTH = {hop_length}")
        print(f"3: WINDOW_TYPE = {window_type}")
        if synthesis_mode_str == 'griffin_lim':
            print(f"4: GRIFFIN_ITER = {griffin_iter}")
        if synthesis_mode_str == 'model_based':
            print(f"4: PEAK_THRESHOLD_RATIO = {peak_threshold_ratio}")
            print(f"5: MAX_PARTIALS = {max_partials}")
            print(f"6: F0_RANGE = {f0_range}")
            print(f"7: TREM_MIN_FREQ = {trem_min_freq}")
            print(f"8: TREM_MAX_FREQ = {trem_max_freq}")

        while True:
            param_choice = input("Enter parameter number to modify (or press Enter to continue): ").strip()
            if param_choice == '':
                break
            elif param_choice == '1':
                try:
                    n_fft = int(input(f"Enter new N_FFT value (current: {n_fft}): "))
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif param_choice == '2':
                try:
                    hop_length = int(input(f"Enter new HOP_LENGTH value (current: {hop_length}): "))
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif param_choice == '3':
                window_type = input(f"Enter new WINDOW_TYPE (current: {window_type}): ")
            elif param_choice == '4':
                if synthesis_mode_str == 'griffin_lim':
                    try:
                        griffin_iter = int(input(f"Enter new GRIFFIN_ITER value (current: {griffin_iter}): "))
                    except ValueError:
                        print("Invalid input. Please enter an integer.")
                elif synthesis_mode_str == 'model_based':
                    try:
                        peak_threshold_ratio = float(input(f"Enter new PEAK_THRESHOLD_RATIO value (current: {peak_threshold_ratio}): "))
                    except ValueError:
                        print("Invalid input. Please enter a float.")
                else:
                    print("Invalid parameter choice for this mode.")
            elif param_choice == '5' and synthesis_mode_str == 'model_based':
                try:
                    max_partials = int(input(f"Enter new MAX_PARTIALS value (current: {max_partials}): "))
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif param_choice == '6' and synthesis_mode_str == 'model_based':
                try:
                    f0_range_str = input(f"Enter new F0_RANGE (e.g., 50.0,2000.0) (current: {f0_range}): ")
                    f0_range = tuple(map(float, f0_range_str.split(',')))
                except ValueError:
                    print("Invalid input. Please enter two comma-separated floats.")
            elif param_choice == '7' and synthesis_mode_str == 'model_based':
                try:
                    trem_min_freq = float(input(f"Enter new TREM_MIN_FREQ value (current: {trem_min_freq}): "))
                except ValueError:
                    print("Invalid input. Please enter a float.")
            elif param_choice == '8' and synthesis_mode_str == 'model_based':
                try:
                    trem_max_freq = float(input(f"Enter new TREM_MAX_FREQ value (current: {trem_max_freq}): "))
                except ValueError:
                    print("Invalid input. Please enter a float.")
            else:
                print("Invalid parameter number.")


    print("Performing STFT analysis...")
    D = stft_analysis(y, n_fft, hop_length, window_type)
    magnitude = np.abs(D)

    print("Extracting selected descriptors...")
    descriptors, phase = extract_all_descriptors(D, sr, hop_length, n_fft)

    # Print some results (same as before)
    print("\n--- GLOBAL DESCRIPTORS ---")
    print(f" Attack Time: {descriptors['attack_time']:.3f}s")
    print(f" Decay Time: {descriptors['decay_time']:.3f}s")
    print(f" Release Time: {descriptors['release_time']:.3f}s")
    print(f" Log Attack Time: {descriptors['log_attack_time']:.3f}")
    print(f" Attack Slope: {descriptors['attack_slope']:.3f}")
    print(f" Decrease Slope: {descriptors['decrease_slope']:.3f}")
    print(f" Effective Duration: {descriptors['effective_duration']:.3f}s")
    print(f" Tremolo Amp: {descriptors['tremolo_amplitude']:.3f}, Freq: {descriptors['tremolo_frequency']:.3f} Hz")

    print("\n--- TIME-VARYING DESCRIPTORS (showing means) ---")
    print(f" Centroid (mean): {np.mean(descriptors['spectral_centroid']):.2f} Hz")
    print(f" Spread (mean): {np.mean(descriptors['spectral_spread']):.2f} Hz")
    print(f" Rolloff (mean): {np.mean(descriptors['spectral_rolloff']):.2f} Hz")
    print(f" Flatness (mean): {np.mean(descriptors['spectral_flatness']):.3f}")
    print(f" Crest (mean): {np.mean(descriptors['spectral_crest']):.3f}")
    print(f" Variation (flux) (mean): {np.mean(descriptors['spectral_variation']):.3f}")
    print(f" Skewness (mean): {np.mean(descriptors['spectral_skewness']):.3f}")
    print(f" Kurtosis (mean): {np.mean(descriptors['spectral_kurtosis']):.3f}")

    print("\n--- PARTIAL-BASED DESCRIPTORS (showing means) ---")
    print(f" f0_series (mean): {np.nanmean(descriptors['f0_series']):.2f} Hz")
    print(f" Inharmonicity (mean): {np.mean(descriptors['inharmonicity']):.3f}")
    print(f" Odd/Even Ratio (mean): {np.mean(descriptors['odd_even_ratio']):.3f}")
    print(f" Noiseness (mean): {np.mean(descriptors['noiseness']):.3f}")
    print(f" Harmonic Deviation (mean): {np.mean(descriptors['harmonic_dev']):.3f}")

    print("\nSynthesis mode:", synthesis_mode_str)
    if synthesis_mode_str == 'direct':
        # Reconstruct from full STFT
        y_syn = synthesis_direct(D)
    elif synthesis_mode_str == 'griffin_lim':
        # Griffin-Lim from magnitude
        y_syn = synthesis_griffin_lim(magnitude, sr, n_fft, hop_length, window_type, griffin_iter)
    elif synthesis_mode_str == 'model_based':
        # Model-based sinusoidal
        y_syn = synthesis_model_based(magnitude, sr, n_fft, hop_length, window_type)
    else:
        raise ValueError("Invalid SYNTHESIS_MODE. Choose 'direct', 'griffin_lim', or 'model_based'.")

    out_file = f'output_{synthesis_mode_str}.wav'
    sf.write(out_file, y_syn, sr)
    print(f"Synthesized audio saved to {out_file}")

    # Optional plotting
    import math
    import librosa.display

    plt.figure(figsize=(12, 12)) # Increased figure height to accommodate more subplots

    # Waveforms
    plt.subplot(4, 1, 1) # 4 rows now
    librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Original')
    librosa.display.waveshow(y_syn, sr=sr, color='r', alpha=0.5, label='Synth')
    plt.legend()
    plt.title("Original vs Synthesized Waveform")

    # Plot Spectral Descriptors
    frames = np.arange(descriptors['spectral_centroid'].shape[0])
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    plt.subplot(4, 1, 2) # Second row
    plt.plot(times, descriptors['spectral_centroid'], label='Centroid')
    plt.plot(times, descriptors['spectral_spread'], label='Spread') # Added Spread
    plt.plot(times, descriptors['spectral_rolloff'], label='Rolloff')
    plt.plot(times, descriptors['spectral_flatness'], label='Flatness') # Added Flatness
    plt.xlabel('Time (s)')
    plt.ylabel('Hz/Ratio') # Adjusted y-label
    plt.title("Spectral Descriptors") # More general title
    plt.legend()

    # Plot Partial-based descriptors
    plt.subplot(4, 1, 3) # Third row
    if descriptors['noiseness'].shape[0] == frames.shape[0]:
        plt.plot(times, descriptors['noiseness'], label='Noiseness', color='g')
        plt.plot(times, descriptors['inharmonicity'], label='Inharmonicity', color='purple') # Added Inharmonicity
        plt.ylim([0, 1]) # Keep noiseness within 0-1, inharmonicity can be larger
        plt.title("Partial-based Descriptors") # More general title
        plt.xlabel('Time (s)')
        plt.legend()
    else:
        plt.text(0.1, 0.5, "Partial-based array length mismatch for plotting", color='red')

    # Plot Skewness and Kurtosis
    plt.subplot(4, 1, 4) # Fourth row
    plt.plot(times, descriptors['spectral_skewness'], label='Skewness', color='orange')
    plt.plot(times, descriptors['spectral_kurtosis'], label='Kurtosis', color='brown')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.title("Spectral Skewness and Kurtosis")
    plt.legend()


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()