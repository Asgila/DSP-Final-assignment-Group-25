
######################################################
# Note this testing script is made completly with AI # 
######################################################

import argparse
import os
import sys
import numpy as np
import scipy.io.wavfile as wav

# Add the parent directory to the path so we can import tune.py from the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tune


def plot_tuning_effects(signal, fs, filename, instrument, deviation):
    import matplotlib.pyplot as plt

    f0 = tune.estimate_fundamental_frequency(signal, fs)
    spectral = tune.analyze_harmonics(signal, fs, f0)
    temporal = tune.analyze_temporal_properties(signal, fs)
    preview_length = min(len(signal), int(0.1 * fs))
    time = np.arange(preview_length) / fs
    frequencies = np.fft.rfftfreq(len(signal), d=1 / fs)
    magnitude = np.abs(np.fft.rfft(signal))
    spectrum_limit = min(5000, fs / 2)

    figure, (waveform_axis, spectrum_axis) = plt.subplots(2, 1, figsize=(11, 7))
    waveform_axis.plot(time, signal[:preview_length], label="Waveform", alpha=0.65)
    waveform_axis.plot(
        time, temporal["envelope"][:preview_length], label="Amplitude envelope", linewidth=2
    )
    waveform_axis.set(
        title=f"{filename}: {instrument} (pitch deviation {deviation:.1f} cents)",
        xlabel="Time (s)",
        ylabel="Amplitude",
    )
    waveform_axis.legend()
    waveform_axis.grid(True, alpha=0.3)

    spectrum_axis.plot(frequencies, magnitude, label="Magnitude spectrum")
    centroid = spectral["spectral_centroid"]
    if centroid <= spectrum_limit:
        spectrum_axis.axvline(
            centroid,
            color="orange",
            linestyle="--",
            label=f"Spectral centroid ({centroid:.0f} Hz)",
        )
    else:
        spectrum_axis.text(
            0.99,
            0.95,
            f"Spectral centroid: {centroid:.0f} Hz (outside view)",
            transform=spectrum_axis.transAxes,
            ha="right",
            va="top",
        )
    if f0 is not None and f0 <= spectrum_limit:
        spectrum_axis.axvline(f0, color="green", linestyle=":", label=f"Fundamental ({f0:.1f} Hz)")
    for index, harmonic in enumerate(spectral["harmonics"], start=1):
        if harmonic <= spectrum_limit:
            spectrum_axis.axvline(
                harmonic,
                color="red",
                alpha=0.35,
                linestyle=":",
                label="Expected harmonics" if index == 1 else None,
            )
    spectrum_axis.set(
        title="Frequency spectrum and tuning features",
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        xlim=(0, spectrum_limit),
    )
    spectrum_axis.legend()
    spectrum_axis.grid(True, alpha=0.3)
    figure.tight_layout()


def test_tuning_pipeline(show_plots=False):
    # Map the filenames to the expected classification strings you set up
    expected_results = {
        'piano.wav': 'Piano',
        'guitar.wav': 'Guitar',
        'violin.wav': 'Violin',
        'trumpet.wav': 'Trumpet'
    }

    # Assuming you run this script from the root DSP-Final-assignment-Group-25 folder
    folder = 'testing/instrument_sounds/' 
    
    print(f"{'File':<15} | {'Expected':<10} | {'Predicted':<10} | {'Deviation (Cents)':<20} | {'Status'}")
    print("-" * 75)
    
    passed_tests = 0

    for filename, expected_instrument in expected_results.items():
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            print(f"{filename:<15} | FILE NOT FOUND at {filepath}")
            continue
            
        fs, signal = wav.read(filepath)
        
        # Convert to mono and normalize to prevent math errors
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        signal = signal.astype(float) / np.max(np.abs(signal))
        
        # Run the full tuning pipeline
        instrument, deviation = tune.process_tuning_pipeline(signal, fs)
        
        # Check if the predicted instrument matches the expected one
        if instrument == expected_instrument:
            status = "PASS"
            passed_tests += 1
        else:
            status = "FAIL"
            
        print(f"{filename:<15} | {expected_instrument:<10} | {instrument:<10} | {deviation:<20.2f} | {status}")
        if show_plots:
            plot_tuning_effects(signal, fs, filename, instrument, deviation)

    print("-" * 75)
    print(f"Test Summary: {passed_tests}/{len(expected_results)} passed.")
    if show_plots:
        import matplotlib.pyplot as plt

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the tuning pipeline.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show waveform, envelope, spectrum, and tuning-feature plots",
    )
    args = parser.parse_args()
    test_tuning_pipeline(show_plots=args.plot)
