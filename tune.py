import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


# Takes in a signal and its sampling frequency, and returns the estimated fundamental frequency using autocorrelation.
def estimate_fundamental_frequency(signal, fs):
    process_length = min(len(signal), fs) 
    short_signal = signal[:process_length]

    #Compute the autocorrelation of the signal
    autocorr = np.correlate(short_signal, short_signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    # Find the peaks in the autocorrelation
    min_lag = int(fs / 2000)  # Minimum lag corresponding to 2 kHz
    max_lag = int(fs / 50)    # Maximum lag corresponding to 50 Hz

    peaks, _ = find_peaks(autocorr[min_lag:max_lag])

    if len(peaks) == 0:
        return None  # No peaks found

    lag = peaks[np.argmax(autocorr[min_lag:max_lag][peaks])] + min_lag
    f0 = fs / lag
    return f0



# Takes in a signal and its sampling frequncy, and returns the spectral centroid and harmonic peaks.
def analyze_harmonics(signal, fs, f0):
    N = len(signal)
    # Compute the FFT of the real part of the signal
    fft_spectrum = np.fft.rfft(signal)
    magnitude_spectrum = np.abs(fft_spectrum)
    frequencies = np.fft.rfftfreq(N, d=1/fs)

    # Calculating the spectral centroid (Center of mass of the spectrum)
    spectral_centroid = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)

    # Find harmonic peaks
    # We search for the peaks in the magnitude spectrum near multiples of the fundamental frequency f0
    harmonics = []
    for i in range(1,5):
        target_freq = i * f0
        harmonics.append(target_freq)
    return {"spectral_centroid": spectral_centroid, "harmonics": harmonics}
 


# takes in a signal and its sampling frequency, and returns the zero crossing rate, attack time, and envelope of the signal.
def analyze_temporal_properties(signal, fs):
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.abs(np.diff(np.sign(signal))))/(2 * len(signal)))

    # taking the envelope extractionx
    rectified_signal = np.abs(signal)

    # Applying a low-pass filter to get the envelope
    nyquist = 0.5 * fs
    cutoff = 10
    b, a = butter(1, cutoff / nyquist, btype='low')
    envelope = filtfilt(b, a, rectified_signal)

    # Estimate the attack time
    max_amp=np.max(envelope)
    start_index = np.where(envelope >= 0.1 * max_amp)
    peak_index = np.argmax(envelope >= 0.9 * max_amp)
    attack_time = (peak_index - start_index[0][0]) / fs 

    return {"zero_crossing_rate": zero_crossings, 
            "attack_time": attack_time, 
            "envelope": envelope}


# Takes in a feature vector and classifies the instrument based on the features.
def classify_instrument(features):
    zcr = features.get("zero_crossing_rate")
    centroid = features.get("spectral_centroid")
    attack = features.get("attack_time") 

    if attack < 0.06:
        # Fast attack, guitar or piano
        if zcr > 0.1:
            return "Guitar"
        else:
            return "Piano"
    else:
        # Slow attack, violin or trumpet
        if centroid > 2000:  # Violin centroid is higher (~2436)
            return "Violin"
        else:
            return "Trumpet"




# Takes in a detected fundamental frequency and calculates how far it deviates from the fundamental frequency of 440 Hz (A4).
def calculate_pitch_deviation(detected_f0, target_freq=440.0):
    # Calculating how far the pitch deviates from the fundamental frequency of 440 Hz (A4)
    if detected_f0 <= 0:
        return 0.0

    # calculate the deviation in cents
    cents_deviation = 1200 * np.log2(detected_f0 / target_freq)

    return cents_deviation
        

def process_tuning_pipeline(signal, fs):
    f0 = estimate_fundamental_frequency(signal, fs)
    deviation = calculate_pitch_deviation(f0, target_freq=440.0)

    spectral_features = analyze_harmonics(signal, fs, f0)
    temporal_features = analyze_temporal_properties(signal, fs)

    # compile the feature vector 

    feature_vector = {
        "f0": f0,
        "spectral_centroid": spectral_features["spectral_centroid"],
        "zero_crossing_rate": temporal_features["zero_crossing_rate"],
        "attack_time": temporal_features["attack_time"],
    }

    instrument = classify_instrument(feature_vector)

    return instrument, deviation


'''
import os
import scipy.io.wavfile as wav


# Assuming the functions from tune.py are imported
files = ['piano.wav', 'guitar.wav', 'violin.wav', 'trumpet.wav']
folder = 'testing/instrument_sounds/'

print(f"{'Instrument':<10} | {'Attack (s)':<10} | {'Centroid (Hz)':<15} | {'Zero-Crossing':<15}")
print("-" * 55)

for file in files:
    fs, signal = wav.read(os.path.join(folder, file))

    # Convert to mono if it's stereo
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    # Convert to float and normalize
    signal = signal.astype(float) / np.max(np.abs(signal))


    # Normalize signal if needed, then extract:
    f0 = estimate_fundamental_frequency(signal, fs)
    temporal = analyze_temporal_properties(signal, fs)
    spectral = analyze_harmonics(signal, fs, f0)
    
    print(f"{file[:-4]:<10} | {temporal['attack_time']:<10.4f} | {spectral['spectral_centroid']:<15.2f} | {temporal['zero_crossing_rate']:<15.4f}")
'''