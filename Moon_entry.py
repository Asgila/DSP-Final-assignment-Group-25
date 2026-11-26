
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def prewarp_frequency(f, fs): #reused prewarp function from Bandpass filter
    """
    Pre-warp the digital frequency f and convert it to angular frequency w (rad/s).
    Returns pre-warped angular frequency (wc) in rad/s.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not (0 < f < fs / 2):
        raise ValueError("Frequency must be between 0 and Nyquist frequency.")

    Ts = 1.0 / fs  # Sampling period
    w = 2 * np.pi * f  # Digital angular frequency in rad/s

    # Pre-warping 
    pre_warped_w = 2/Ts * np.tan(w * Ts / 2)

    return pre_warped_w

cutoff = 0.25 
ripple_db = 1.5 #large ripple for better effect
order = 5
bits = 8

dropout_duration = 0.1
drop_progress = 0.2
ramp_s = 0.005


def chebychev_filter(cutoff, ripple_db, order, fs):
    #same algorithm as Bandbass filter

    b_analog, a_analog = signal.iirfilter(order, Wn=1.0, rp=ripple_db, analog=True, ftype='cheby1', btype='low')

    #prewarp the digital cutoff
    wc = prewarp_frequency(cutoff, fs)
    b_analog, a_analog = signal.lp2lp(b_analog, a_analog, wo=wc)
    #bilinear transform(s to z)
    b, a = signal.bilinear(b_analog, a_analog, fs=fs)
    return b, a #filter coefficents
def is_stable(a):
    return np.all(np.abs(np.roots(a))<1.0) #is stable if all poles are inside the unit circle

from blackout import simulate_quantization_reduction #reused function from  blackout will be used to round the signal to 2^bits levels

def dropout(duration,ramp,window_type = 'hann'):
    ramp = min(ramp, duration // 2)
    window = signal.windows.get_window(window_type, 2 * ramp, fftbins=False)  # symmetric window
    drop = np.zeros(duration)
    drop[:ramp] = window[ramp:] #falling half(1 to 0)
    drop[duration - ramp:] = window[:ramp] #rising half (0 to 1)
    return drop

def masking(N,fs,dropouts): #makes dropout with gain at certain points
    mask = np.ones(N)
    T_total = N / fs
    ramp = int(ramp_s * fs)
    st = T_total * np.sqrt((np.arange(dropouts) + 0.5)/dropouts) #sqrt gives more droputs as density scales linearly with time
    for t in st:
        progress = t/T_total
        duration = int((dropout_duration + drop_progress * progress**2) * fs) #dropots get longer
        start = int(t * fs)
        stop = min(start + duration,N)
        area = dropout(duration,ramp)[:stop -start]
        mask[start:stop] = np.minimum(mask[start:stop],area) #keeps the deeper one
    return mask

def moon_shadow(x,fs,dropouts = 16, T = None, bits = bits):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    T = N / fs if T is None else T
    b, a = chebychev_filter(cutoff * fs, ripple_db, order, fs)
    if not is_stable(a):
        raise RuntimeError("Chebyshev filter is unstable")
    filtered = signal.filtfilt(b, a, x, axis=0)

    degraded = filtered if bits is None else simulate_quantization_reduction(filtered, target_bit_depth=bits)

    #fade
    t = np.arange(N) /fs #time vector
    fade = np.clip(t / T,0,1)
    transition = (1 - fade) * x + fade * degraded
    mask = masking(N,fs,dropouts)
    output = transition * mask
    return output, (b, a)