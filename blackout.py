import numpy as np
from scipy.interpolate import interp1d, CubicSpline


# This function takes in a signal and interpolates over the dropout regions to fill in the missing samples.
def simulate_quantization_reduction(signal, original_bit_depth=16, target_bit_depth=8):
    # simulate quantization reduction by scaling the signal to the target bit depth

    max_val=np.max(np.abs(signal))
    if max_val == 0:
        return signal  # Avoid division by zero for silent signals
    normalized_signal = signal / max_val  # Normalize to [-1, 1]

    # Calculate the quantization levels for the target bit depth
    quantization_levels = 2 ** target_bit_depth

    # scale to new range, round to nearest integer, and scale back to [-1, 1]
    scaled_signal = normalized_signal * (quantization_levels / 2 - 0.5)
    quantized_signal = np.round(scaled_signal)
    dequantized_signal = quantized_signal / (quantization_levels / 2 - 0.5)

    # Scale back to original amplitude
    output_signal = dequantized_signal * max_val

    return output_signal



# This function takes in a signal and interpolates over the dropout regions to fill in the missing samples.
def interpolate_dropout(signal, fs, start_time, duration=2.0, method='cubic'):
    # Perform interpolation over the dropout region
    # Requires 2 seconds of dropout

    # Calculate the sample indecies
    start_sample = int(start_time * fs)
    num_dropout_samples = int(duration * fs)
    end_sample = start_sample + num_dropout_samples

    # ensure indexes are within the signal bounds
    if end_sample > len(signal):
        end_sample = len(signal)-1
        num_dropout_samples = end_sample - start_sample

    # Create a copy of the signal to modify
    reconstructed_signal = np.copy(signal)

    # Get the samples before and after the dropout region
    context_samples = int(0.05 * fs)  # 50 ms of context on each side

    pre_gap_x = np.arange(start_sample - context_samples, start_sample)
    post_gap_x = np.arange(end_sample, end_sample + context_samples)

    # Combine the valid x (time/indecies) and y (amplitude) values 
    valid_x = np.concatenate((pre_gap_x, post_gap_x))
    valid_y = np.concatenate((signal[pre_gap_x], signal[post_gap_x]))

    # The x values we want to predict (the dropout region)
    dropout_x = np.arange(start_sample, end_sample)

    # Perform interpolation based on the specified method
    if method == 'linear':
        interpolator = interp1d(valid_x, valid_y, kind='linear')
        reconstructed_signal[start_sample:end_sample] = interpolator(dropout_x)
    elif method == 'cubic':
        interpolator = interp1d(valid_x, valid_y, kind='cubic')
        reconstructed_signal[start_sample:end_sample] = interpolator(dropout_x)
    elif method == 'spline':
        interpolator = CubicSpline(valid_x, valid_y)
        reconstructed_signal[start_sample:end_sample] = interpolator(dropout_x)
    else:
        raise ValueError("Unsupported method. Choose 'linear', 'cubic', or 'spline'.")

    return reconstructed_signal


# This function calculates the Signal-to-Noise Ratio (SNR) between the original and degraded signals.
def calculate_snr(original_signal, degraded_signal):
    noise = original_signal - degraded_signal
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')  # No noise, infinite SNR

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


# This function applies a complete blackout phase to the signal, simulating a dropout and attempting to reconstruct it.
def apply_complete_blackout_phase(signal, fs, blackout_start_time):
    
    # simulate reduced quantization before the dropout
    degraded_signal = simulate_quantization_reduction(signal, target_bit_depth=4)

    # simulate the 2 second dropout (total signal loss)
    start_idx=int(blackout_start_time * fs)
    end_idx=start_idx + int(2.0 * fs)
    blackout_signal = np.copy(degraded_signal)
    blackout_signal[start_idx:end_idx] = 0.0

    # Attemt to reconstruct the signal using interpolation
    reconstructed_signal = interpolate_dropout(
        blackout_signal, 
        fs, 
        start_time=blackout_start_time, 
        duration=2.0, 
        method='spline'
    )

    return blackout_signal, reconstructed_signal
    