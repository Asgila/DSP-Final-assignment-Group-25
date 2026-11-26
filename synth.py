"""Synthesizer and clean transmission generation module (from Synthesizer_Clean_Transmission.ipynb)."""

import math
import os
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

DEFAULT_FS = 44100

OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTALS = {"#": 1, "b": -1, "": 0}
VOICINGS_NAMES = {
    "F": ["F3", "A3", "C4"],
    "C": ["E3", "G3", "C4"],
    "G": ["D3", "G3", "B3"],
    "Am": ["C3", "E3", "A3"],
    "C#m": ["E3", "G#3", "C#4"],
    "Bb": ["D3", "F3", "Bb3"],
}

SHORT_BARS = [
    ("F", ["A4", "G4", "F4", "G4"]),
    ("F", ["A4", "A4", "A4:2"]),
    ("C", ["G4", "G4", "G4:2"]),
    ("F", ["A4", "C5", "C5:2"]),
    ("F", ["A4", "G4", "F4", "G4"]),
    ("F", ["A4", "A4", "A4", "A4"]),
    ("C", ["G4", "G4", "A4", "G4"]),
    ("F", ["F4:4"]),
]

FULL_BARS = [
    # Verse
    ("F", ["C4", "A4", "G4", "F4"]),
    ("F", ["C4:3", "C4:0.5", "C4:0.5"]),
    ("F", ["C4", "A4", "G4", "F4"]),
    ("Bb", ["D4:4"]),
    ("Bb", ["D4", "Bb4", "A4", "G4"]),
    ("C", ["E4:4"]),
    ("C", ["C5", "C5", "Bb4", "G4"]),
    ("F", ["A4:4"]),
    ("F", ["C4", "A4", "G4", "F4"]),
    ("F", ["C4:4"]),
    ("F", ["C4", "A4", "G4", "F4"]),
    ("Bb", ["D4:4"]),
    ("Bb", ["D4", "Bb4", "A4", "G4"]),
    ("C", ["C5", "C5", "C5", "C5"]),
    ("C", ["D5", "C5", "Bb4", "G4"]),
    ("F", ["F4:4"]),
    # Chorus
    ("F", ["A4", "A4", "A4:2"]),
    ("F", ["A4", "A4", "A4:2"]),
    ("F", ["A4", "C5", "F4:1.5", "G4:0.5"]),
    ("F", ["A4:4"]),
    ("Bb", ["Bb4", "Bb4", "Bb4:1.5", "Bb4:0.5"]),
    ("F", ["Bb4", "A4", "A4", "A4:0.5", "A4:0.5"]),
    ("C", ["A4", "G4", "G4", "A4"]),
    ("C", ["G4:2", "C5:2"]),
    ("F", ["A4", "A4", "A4:2"]),
    ("F", ["A4", "A4", "A4:2"]),
    ("F", ["A4", "C5", "F4:1.5", "G4:0.5"]),
    ("F", ["A4:4"]),
    ("Bb", ["Bb4", "Bb4", "Bb4:1.5", "Bb4:0.5"]),
    ("F", ["Bb4", "A4", "A4", "A4:0.5", "A4:0.5"]),
    ("C", ["C5", "C5", "Bb4", "G4"]),
    ("F", ["F4:4"]),
]


def generate_sinusoid(freq, sample_rate=DEFAULT_FS, duration=1.0):
    t = np.arange(0, duration, 1 / sample_rate)
    return t, np.sin(2 * np.pi * freq * t)


def generate_square_wave(freq, sample_rate=DEFAULT_FS, duration=1.0, duty_cycle=0.5):
    t = np.arange(0, duration, 1 / sample_rate)
    phase = (t * freq) % 1
    return t, np.where(phase < duty_cycle, 1.0, -1.0)


def generate_triangle_wave(freq, sample_rate=DEFAULT_FS, duration=1.0):
    t, square_wave = generate_square_wave(freq, sample_rate, duration)
    triangle_wave = np.cumsum(square_wave) / sample_rate
    triangle_wave = 4 * freq * triangle_wave - 1.0
    return t, triangle_wave


def generate_sawtooth_wave(freq, sample_rate=DEFAULT_FS, duration=1.0):
    t = np.arange(0, duration, 1 / sample_rate)
    phase = (freq * t) % 1
    return t, 2.0 * phase - 1.0


def _exp_segment(start, end, n_samples, factor):
    if n_samples <= 0:
        return np.empty(0, dtype=float)
    x = np.linspace(0.0, 1.0, n_samples)
    if factor == 0:
        return np.linspace(start, end, n_samples)
    m = (1.0 - np.exp(-factor * x)) / (1.0 - np.exp(-factor))
    return start + (end - start) * m


def adsr_envelope(
    a,
    d,
    s,
    r,
    duration,
    fs=DEFAULT_FS,
    attack_exponential_factor=0,
    decay_exponential_factor=0,
    release_exponential_factor=0,
):
    """Build an ADSR amplitude envelope of total length int(duration * fs)."""
    total_samples = int(duration * fs)
    attack_samples = int(a * fs)
    decay_samples = int(d * fs)
    release_samples = min(int(r * fs), total_samples)
    sustain_samples = max(0, total_samples - (attack_samples + decay_samples + release_samples))

    attack = _exp_segment(0.0, 1.0, attack_samples, attack_exponential_factor)
    decay = _exp_segment(1.0, s, decay_samples, decay_exponential_factor)
    sustain = np.full(sustain_samples, s, dtype=float)
    release = _exp_segment(s, 0.0, release_samples, release_exponential_factor)
    return np.concatenate((attack, decay, sustain, release))[:total_samples]


def lowpass_filter(input_signal, cutoff_f, fs=DEFAULT_FS, filter_type="butter", order=2):
    """Apply an IIR low-pass filter, clamping cutoff strictly below Nyquist."""
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff_f / nyquist, 0.99)
    if filter_type == "butter":
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    elif filter_type == "cheby1":
        b, a = signal.cheby1(order, 0.5, normal_cutoff, btype="low", analog=False)
    elif filter_type == "cheby2":
        b, a = signal.cheby2(order, 20, normal_cutoff, btype="low", analog=False)
    elif filter_type == "elliptical":
        b, a = signal.ellip(order, 0.5, 20, normal_cutoff, btype="low", analog=False)
    else:
        raise ValueError("Unsupported filter type.")
    return signal.lfilter(b, a, input_signal), b, a


def all_pass_filter(input_signal, fc, fs=DEFAULT_FS, order=1):
    """Cascade first-order all-pass filters for phase dispersion/ambience."""
    coeff = (math.tan(math.pi * fc / fs) - 1) / (math.tan(math.pi * fc / fs) + 1)
    b = np.array([coeff, 1.0])
    a = np.array([1.0, coeff])
    filtered = input_signal
    for _ in range(order):
        filtered = signal.lfilter(b, a, filtered)
    return filtered, b, a


def name_to_midi(name):
    letter = name[0]
    rest = name[1:]
    if rest and rest[0] in ("#", "b"):
        acc, octave = rest[0], int(rest[1:])
    else:
        acc, octave = "", int(rest)
    return 12 * (octave + 1) + OFFSETS[letter] + ACCIDENTALS[acc]


def midi_to_freq(midi_note):
    return 440.0 * 2.0 ** ((midi_note - 69) / 12.0)


def parse_bars(bars, beats_per_bar=4):
    voicings = {k: [name_to_midi(n) for n in v] for k, v in VOICINGS_NAMES.items()}
    melody, chords = [], []
    for i, (chord, notes) in enumerate(bars):
        bar_start = i * beats_per_bar
        for midi_note in voicings[chord]:
            chords.append((midi_note, bar_start, beats_per_bar))
        t = bar_start
        for token in notes:
            name, _, length = token.partition(":")
            dur = float(length) if length else 1.0
            melody.append((name_to_midi(name), t, dur))
            t += dur
    return melody, chords


def render_note(midi, hold_time, wave, a, d, s, r, cutoff_mult, fs=DEFAULT_FS, **env_kwargs):
    freq = midi_to_freq(midi)
    if a + d > hold_time and (a + d) > 0:
        scale = hold_time / (a + d)
        a, d = a * scale, d * scale

    total_time = hold_time + r
    adsr = adsr_envelope(a, d, s, r, duration=total_time, fs=fs, **env_kwargs)
    _, note_waveform = wave(freq, fs, len(adsr) / fs)
    note_waveform = note_waveform[: len(adsr)]
    note_waveform = lowpass_filter(note_waveform, freq * cutoff_mult, fs=fs)[0]
    return note_waveform * adsr[: len(note_waveform)]


def render_track(notes, bpm=120, fs=DEFAULT_FS, **voice_params):
    seconds_per_beat = 60.0 / bpm
    last_end = max((st + ln) * seconds_per_beat for _, st, ln in notes) + 2.0
    out = np.zeros(int(last_end * fs), dtype=float)
    for midi, start_beat, len_beats in notes:
        note = render_note(midi, len_beats * seconds_per_beat, fs=fs, **voice_params)
        idx = int(start_beat * fs * seconds_per_beat)
        end_idx = min(idx + len(note), len(out))
        out[idx:end_idx] += note[: end_idx - idx]
    return out


def generate_clean_transmission(
    fs=DEFAULT_FS,
    use_full_song=True,
    bpm=120,
    ir_path="USS Dolphin Submarine Control Room.wav",
):
    """Synthesize the clean interstellar radio music with optional submarine IR reverb."""
    bars = FULL_BARS if use_full_song else SHORT_BARS
    melody, chords = parse_bars(bars)
    lead = render_track(
        melody,
        bpm=bpm,
        fs=fs,
        wave=generate_sawtooth_wave,
        a=0.005,
        d=0.15,
        s=0.3,
        r=0.15,
        cutoff_mult=6,
        decay_exponential_factor=3,
    )
    pad = render_track(
        chords,
        bpm=bpm,
        fs=fs,
        wave=generate_triangle_wave,
        a=0.12,
        d=0.1,
        s=0.8,
        r=0.4,
        cutoff_mult=3,
    )
    n = max(len(lead), len(pad))
    lead = np.pad(lead, (0, n - len(lead)))
    pad = np.pad(pad, (0, n - len(pad)))
    mixed = 0.7 * lead + 0.3 * pad

    allpassed = all_pass_filter(mixed, 800, fs, order=6)[0]
    allpass_mixed = 0.5 * mixed + 0.5 * allpassed
    peak = np.max(np.abs(allpass_mixed))
    if peak > 0:
        allpass_mixed = allpass_mixed / peak

    if ir_path and os.path.exists(ir_path):
        fs_ir, ir = wavfile.read(ir_path)
        ir = ir.astype(float)
        if ir.ndim == 2:
            ir = ir.mean(axis=1)
        if fs_ir != fs:
            from math import gcd

            g = gcd(int(fs), int(fs_ir))
            ir = signal.resample_poly(ir, int(fs) // g, int(fs_ir) // g)
        ir_sum = np.sum(np.abs(ir))
        if ir_sum > 0:
            ir = ir / ir_sum
        allpass_mixed = signal.fftconvolve(allpass_mixed, ir)

    peak = np.max(np.abs(allpass_mixed))
    return allpass_mixed / peak if peak > 0 else allpass_mixed


def generate_detection_beep(fs=DEFAULT_FS, duration=0.45):
    """Create a brief two-tone scanner confirmation chirp for Scene 2."""
    half = duration / 2.0
    _, t1 = generate_sinusoid(880.0, fs, half)
    _, t2 = generate_sinusoid(1320.0, fs, half)
    env1 = adsr_envelope(0.01, 0.04, 0.7, 0.05, len(t1) / fs, fs=fs)
    env2 = adsr_envelope(0.01, 0.04, 0.7, 0.08, len(t2) / fs, fs=fs)
    n1 = min(len(t1), len(env1))
    n2 = min(len(t2), len(env2))
    return 0.35 * np.concatenate([t1[:n1] * env1[:n1], t2[:n2] * env2[:n2]])
