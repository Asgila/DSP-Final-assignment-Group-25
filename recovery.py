"""Audio I/O, resampling, cross-fading, step-wise recovery, and stereo mastering."""

from math import gcd
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

from blackout import simulate_quantization_reduction


def load_audio(path, target_fs=44100):
    """
    Load a WAV file, convert to mono float64 in [-1, 1], and resample to target_fs
    using polyphase anti-aliasing filtering when sample rates differ.
    """
    fs_in, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.integer):
        max_int = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float64) / max_int
    else:
        audio = data.astype(np.float64)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if fs_in != target_fs:
        g = gcd(int(target_fs), int(fs_in))
        up = int(target_fs) // g
        down = int(fs_in) // g
        audio = signal.resample_poly(audio, up, down)

    return target_fs, audio


def fit_duration(x, fs, duration_s, edge_fade_s=0.02):
    """Trim or repeat-pad an audio clip to exact duration_s with smooth edge fades."""
    target_n = int(round(duration_s * fs))
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or target_n <= 0:
        return np.zeros(max(0, target_n), dtype=float)

    if len(x) < target_n:
        reps = int(np.ceil(target_n / len(x)))
        x = np.tile(x, reps)
    out = x[:target_n].copy()

    fade_n = min(int(edge_fade_s * fs), target_n // 4)
    if fade_n > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_n) / fade_n))
        out[:fade_n] *= ramp
        out[-fade_n:] *= ramp[::-1]
    return out


def resample_bandwidth_step(x, fs, low_fs):
    """Downsample to `low_fs` (with anti-aliasing) and upsample back to `fs` to model reduced bandwidth."""
    if low_fs >= fs:
        return x.copy()
    g = gcd(int(low_fs), int(fs))
    down = signal.resample_poly(x, int(low_fs) // g, int(fs) // g)
    up = signal.resample_poly(down, int(fs) // g, int(low_fs) // g)
    if len(up) < len(x):
        up = np.pad(up, (0, len(x) - len(up)))
    return up[: len(x)]


def crossfade_concat(segments, fs, crossfade_s=0.15):
    """
    Concatenate audio segments using raised-cosine (Hann) cross-fades to prevent clicks.
    Each item in `segments` may be an array or a tuple `(array, fade_in_s)`.
    """
    if not segments:
        return np.zeros(0, dtype=float)

    first = segments[0][0] if isinstance(segments[0], tuple) else segments[0]
    out = np.asarray(first, dtype=float).copy()

    for item in segments[1:]:
        if isinstance(item, tuple):
            seg, xfade_s = item
        else:
            seg, xfade_s = item, crossfade_s
        seg = np.asarray(seg, dtype=float)
        m = min(int(round(xfade_s * fs)), len(out) // 2, len(seg) // 2)
        if m > 0:
            fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / m))
            fade_out = 1.0 - fade_in
            overlap = out[-m:] * fade_out + seg[:m] * fade_in
            out = np.concatenate([out[:-m], overlap, seg[m:]])
        else:
            out = np.concatenate([out, seg])
    return out


def apply_gradual_recovery(clean_signal, fs=44100, residual_noise=None):
    """
    Simulate emerging from the moon shadow: step-wise restoration of bandwidth
    (via resampling), quantization bit depth, and residual noise attenuation.
    """
    x = np.asarray(clean_signal, dtype=float)
    n = len(x)
    if residual_noise is None:
        rng = np.random.default_rng(123)
        noise = rng.standard_normal(n) * (0.08 * np.sqrt(np.mean(x**2) + 1e-12))
    else:
        noise = np.asarray(residual_noise, dtype=float)[:n]

    # 4 progressive stages: (fraction, effective_fs, bit_depth, noise_scale)
    stages = [
        (0.22, 8000, 4, 0.7),
        (0.26, 16000, 6, 0.35),
        (0.26, 22050, 8, 0.12),
        (0.26, fs, 16, 0.0),
    ]

    pieces = []
    idx = 0
    for i, (frac, eff_fs, bits, n_scale) in enumerate(stages):
        end_idx = n if i == len(stages) - 1 else min(n, idx + int(round(frac * n)))
        chunk = x[idx:end_idx] + n_scale * noise[idx:end_idx]
        chunk = resample_bandwidth_step(chunk, fs, eff_fs)
        if bits < 16:
            chunk = simulate_quantization_reduction(chunk, target_bit_depth=bits)
        pieces.append(chunk)
        idx = end_idx

    return crossfade_concat(pieces, fs=fs, crossfade_s=0.12)[:n]


def master_and_save_stereo(mono_track, out_path, fs=44100, target_peak=0.9):
    """Normalize headroom, duplicate to stereo (2-ch), and export a 16-bit WAV file."""
    mono = np.asarray(mono_track, dtype=float)
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = target_peak * (mono / peak)

    stereo = np.column_stack([mono, mono])
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(out_path, fs, pcm16)
    return stereo
