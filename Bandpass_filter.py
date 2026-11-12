# Bandpass filter (IIR)- Bilinear transform - 
# Analog-to-digital filter conversion via bilinear transform with documented pre-warping calculations (in code) 



# First design an analog lowpass Butterworth filter, using bilinertranform

import numpy as np


def design_analog_lp_prototype(order):
    """
    Designs a normalized analog Butterworth lowpass prototype 
    in the s-domain with cutoff omega_p_hat = 1 rad/s.
    Returns analog polynomial coefficients (b_s, a_s) in descending powers of s.
    """
    if order <= 0:
        raise ValueError("Order must be a positive integer.")

    # Poles on the left-half unit circle in the s-plane
    k = np.arange(order)
    angles = np.pi / 2 + (2 * k + 1) * np.pi / (2 * order)
    s_poles = np.exp(1j * angles)

    # Numerator is 1, Denominator is expanded from the poles
    b_s = np.array([1.0])
    a_s = np.poly(s_poles).real  # [1, a_{N-1}, ..., 1]

    return b_s, a_s

def prewarp_frequency(fc, fs):
    """
    Pre-warp the digital passband frequency fc to angular frequency wc
    Returns pre-warped angular frequency (wc) in rad/s.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    if not (0 < fc < fs / 2):
        raise ValueError("Passband frequency must be between 0 and Nyquist frequency.")

    Ts = 1.0 / fs  # Sampling period
    w_c = 2 * np.pi * fc  # Digital angular frequency in rad/s

    # Pre-warping 
    pre_warped_wc = 2/Ts * np.tan(w_c * Ts / 2)

    return pre_warped_wc

# or just use signal.lp2bp
def move_invert_scale_lp_to_bp(b_s, a_s, wp1, wp2):
    """
    Frequency transform by substitution s -> T(s) = (s^2 + wp1*wp2) / (s*(wp2 - wp1))
    Takes analog prototype (b_s, a_s) and pre-warped analog frequencies wp1, wp2 (rad/s).
    Returns analog bandpass coefficients (b_bp_s, a_bp_s) in descending powers of s.
    """
    N = len(a_s) - 1  # Prototype order
    bw = wp2 - wp1    # Bandwidth: (wp2 - wp1)
    w0_sq = wp1 * wp2 # Center frequency squared: wp1 * wp2

    # Pad b_s with leading zeros so b_s and a_s both have length N + 1
    b_s_padded = np.pad(b_s, (N + 1 - len(b_s), 0))

    # Polynomial building blocks in descending powers of s:
    # Num(s) = s^2 + wp1*wp2  --> [1, 0, w0_sq]
    # Den(s) = s*(wp2 - wp1)  --> [bw, 0]
    Num_poly = np.array([1.0, 0.0, w0_sq])
    Den_poly = np.array([bw, 0.0])

    b_bp_s = np.zeros(2 * N + 1)
    a_bp_s = np.zeros(2 * N + 1)

    for i in range(N + 1):
        power = N - i  # Current power of s
        
        # Compute (s^2 + w0_sq)^power
        num_part = np.array([1.0])
        for _ in range(power):
            num_part = np.convolve(num_part, Num_poly)
            
        # Compute (bw * s)^(N - power)
        den_part = np.array([1.0])
        for _ in range(N - power):
            den_part = np.convolve(den_part, Den_poly)
            
        term = np.convolve(num_part, den_part)
        
        b_bp_s += b_s_padded[i] * term
        a_bp_s += a_s[i] * term

    return b_bp_s, a_bp_s

#signal.bilinear(b, a, fs=1.0) # no prewarping is done in this function
def bilinear_transform(b_s, a_s, fs):
    """
    Substitutes s = (2/Ts) * (1 - z^-1) / (1 + z^-1) into analog polynomials
    to obtain digital filter coefficients (b_z, a_z).
    """
    M = len(a_s) - 1  # Total analog filter order (2 * N)
    Ts = 1.0 / fs
    c = 2.0 / Ts

    b_s_padded = np.pad(b_s, (M + 1 - len(b_s), 0))

    # In powers of z^-1: (1 - z^-1) is [1, -1], and (1 + z^-1) is [1, 1]
    diff_poly = np.array([1.0, -1.0])
    sum_poly = np.array([1.0, 1.0])

    b_z = np.zeros(M + 1)
    a_z = np.zeros(M + 1)

    for i in range(M + 1):
        power = M - i  # Power of s
        
        # (c * (1 - z^-1))^power
        p1 = np.array([1.0])
        for _ in range(power):
            p1 = np.convolve(p1, c * diff_poly)
            
        # (1 + z^-1)^(M - power)
        p2 = np.array([1.0])
        for _ in range(M - power):
            p2 = np.convolve(p2, sum_poly)
            
        term = np.convolve(p1, p2)
        
        b_z += b_s_padded[i] * term
        a_z += a_s[i] * term

    # Normalize by a_z[0] so the leading denominator coefficient is 1.0
    b_z = b_z / a_z[0]
    a_z = a_z / a_z[0]

    return b_z, a_z
