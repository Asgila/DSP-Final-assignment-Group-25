"""Broadband radiation storm noise generation, spectral shaping, and PSD analysis (from Broadband_noise.ipynb)."""

import numpy as np
import scipy.signal as signal


def make_spectrum(x, fs):
    """Compute scaled FFT magnitude spectrum and corresponding frequency axis."""
    n = len(x)
    spectrum = np.fft.fft(x) / n
    freq = np.arange(n) * (fs / n)
    return spectrum, freq


def generate_broadband_noise(n_samples, fs=44100, crackle_rate=10.0, seed=42):
    """
    Generate broadband hiss + impulsive crackle simulating radiation interference near a metallic moon.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)

    # Slight high-mid emphasis (radio static hiss) via 1st-order high-shelf / gentle shaping
    b_shape, a_shape = signal.butter(2, [400.0, 12000.0], btype="band", fs=fs)
    hiss = signal.lfilter(b_shape, a_shape, white)

    # Sparse impulsive crackles filtered softly so they sound like radio static bursts
    crackle_prob = crackle_rate / fs
    impulses = (rng.random(n_samples) < crackle_prob) * rng.uniform(-3.0, 3.0, n_samples)
    b_crk, a_crk = signal.butter(2, 5000.0, btype="low", fs=fs)
    crackle = signal.lfilter(b_crk, a_crk, impulses)

    combined = hiss + 1.5 * crackle
    rms = np.sqrt(np.mean(combined**2))
    return combined / rms if rms > 0 else combined


def compute_psd_welch(x, fs=44100, nperseg=1024, window="hann"):
    """Estimate Power Spectral Density using Welch's averaged periodogram method."""
    return signal.welch(x, fs=fs, window=window, nperseg=nperseg)


def compute_psd_periodogram(x, fs=44100, window="hann"):
    """Estimate Power Spectral Density using a single-window periodogram."""
    return signal.periodogram(x, fs=fs, window=window)


def apply_radiation_storm(clean_signal, fs=44100, snr_db=4.0, ramp_s=1.0, seed=42):
    """
    Overlay broadband static/crackle noise onto the radio feed at a controlled SNR,
    ramping in over `ramp_s` seconds to simulate entering the Chebychev-moon orbit.
    """
    x = np.asarray(clean_signal, dtype=float)
    n = len(x)
    noise = generate_broadband_noise(n, fs=fs, seed=seed)

    sig_rms = np.sqrt(np.mean(x**2))
    if sig_rms == 0:
        sig_rms = 1.0
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))

    ramp_samples = min(n, int(ramp_s * fs))
    env = np.ones(n, dtype=float)
    if ramp_samples > 0:
        env[:ramp_samples] = np.linspace(0.15, 1.0, ramp_samples)

    storm_noise = target_noise_rms * env * noise
    return x + storm_noise, storm_noise
