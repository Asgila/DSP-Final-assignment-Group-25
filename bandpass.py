"""Narrowband anomaly (6-8 kHz) noise, IIR bilinear band-pass/band-stop, and FIR band-stop."""

import numpy as np
import scipy.signal as signal


def prewarp_frequency(f, fs):
    """Pre-warp digital frequency f (Hz) to analog angular frequency (rad/s) for bilinear transform."""
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not (0 < f < fs / 2):
        raise ValueError("Frequency must be between 0 and Nyquist frequency.")

    Ts = 1.0 / fs
    w = 2.0 * np.pi * f
    return (2.0 / Ts) * np.tan(w * Ts / 2.0)


def _design_bilinear_band(band, fs, order, btype):
    f1, f2 = band
    b_proto, a_proto = signal.iirfilter(Wn=1.0, N=order, analog=True, ftype="butter", btype="low")
    w1 = prewarp_frequency(f1, fs)
    w2 = prewarp_frequency(f2, fs)
    w0 = np.sqrt(w1 * w2)
    bw = w2 - w1

    if btype == "bandpass":
        b_analog, a_analog = signal.lp2bp(b_proto, a_proto, w0, bw)
    elif btype == "bandstop":
        b_analog, a_analog = signal.lp2bs(b_proto, a_proto, w0, bw)
    else:
        raise ValueError("btype must be 'bandpass' or 'bandstop'.")

    return signal.bilinear(b_analog, a_analog, fs=fs)


def design_iir_bandpass(passband=(6000.0, 8000.0), fs=44100, order=4):
    """Design an IIR Butterworth bandpass filter via analog prototype + pre-warped bilinear transform."""
    return _design_bilinear_band(passband, fs, order, "bandpass")


def design_iir_bandstop(stopband=(6000.0, 8000.0), fs=44100, order=4):
    """Design an IIR Butterworth band-stop filter via analog prototype + pre-warped bilinear transform."""
    return _design_bilinear_band(stopband, fs, order, "bandstop")


def design_fir_bandstop(stopband=(6000.0, 8000.0), fs=44100, numtaps=401, window="hamming"):
    """Design a linear-phase FIR band-stop filter using the window method (firwin)."""
    if numtaps % 2 == 0:
        numtaps += 1  # Odd length ensures Type I symmetric FIR for band-stop
    b = signal.firwin(numtaps, list(stopband), pass_zero="bandstop", window=window, fs=fs)
    return b, np.array([1.0])


def generate_narrowband_noise(n_samples, fs=44100, band=(6000.0, 8000.0), order=4, seed=42):
    """Generate unit-RMS narrowband noise confined to the specified frequency band."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    b_bp, a_bp = design_iir_bandpass(band, fs=fs, order=order)
    narrow = signal.lfilter(b_bp, a_bp, white)
    rms = np.sqrt(np.mean(narrow**2))
    return narrow / rms if rms > 0 else narrow


def apply_anomaly_phase(
    clean_signal,
    fs=44100,
    band=(6000.0, 8000.0),
    noise_level=0.18,
    order=4,
    fir_numtaps=401,
    cleanup_fraction=0.45,
    filter_type="iir",
    seed=42,
):
    """
    Simulate entering the 6-8 kHz anomaly zone and engaging a band-stop filter to remove it.

    First (1 - cleanup_fraction) of the segment ramps in the 6-8 kHz narrowband interference;
    the remaining cleanup_fraction demonstrates removing the interference with IIR or FIR band-stop.
    """
    x = np.asarray(clean_signal, dtype=float)
    n = len(x)
    noise = generate_narrowband_noise(n, fs=fs, band=band, order=order, seed=seed)

    # Gradual fade-in of the 6-8 kHz hum over the first half, then steady interference
    split = int((1.0 - cleanup_fraction) * n)
    env = np.ones(n, dtype=float)
    ramp_len = max(1, int(0.6 * split))
    env[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

    noisy_signal = x + noise_level * env * noise

    b_iir_bs, a_iir_bs = design_iir_bandstop(band, fs=fs, order=order)
    b_fir_bs, a_fir_bs = design_fir_bandstop(band, fs=fs, numtaps=fir_numtaps)

    if filter_type == "iir":
        cleaned_full = signal.filtfilt(b_iir_bs, a_iir_bs, noisy_signal)
    elif filter_type == "fir":
        cleaned_full = signal.filtfilt(b_fir_bs, a_fir_bs, noisy_signal)
    else:
        raise ValueError("filter_type must be 'iir' or 'fir'.")

    # Smooth transition at `split` from raw noisy anomaly to band-stop-cleaned signal
    xfade = min(int(0.08 * fs), split, n - split)
    output = np.copy(noisy_signal)
    if xfade > 0:
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(xfade) / xfade))
        output[split : split + xfade] = (
            (1.0 - w) * noisy_signal[split : split + xfade]
            + w * cleaned_full[split : split + xfade]
        )
        output[split + xfade :] = cleaned_full[split + xfade :]
    else:
        output[split:] = cleaned_full[split:]

    filters = {
        "iir_bandstop": (b_iir_bs, a_iir_bs),
        "fir_bandstop": (b_fir_bs, a_fir_bs),
    }
    return output, noisy_signal, filters
