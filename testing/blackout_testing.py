
######################################################
# Note this testing script is made completly with AI # 
######################################################


import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import blackout.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import blackout


def plot_blackout_effects():
    import matplotlib.pyplot as plt

    fs = 44100
    duration = 3.0
    time = np.linspace(0, duration, int(fs * duration), endpoint=False)
    original = np.sin(2 * np.pi * 440 * time)
    quantized = blackout.simulate_quantization_reduction(original, target_bit_depth=4)
    blackout_start = 0.5
    blackout_signal, reconstructed = blackout.apply_complete_blackout_phase(
        original, fs, blackout_start
    )
    blackout_end = blackout_start + 2.0

    figure, (quantization_axis, blackout_axis) = plt.subplots(2, 1, figsize=(11, 7))
    sample_window = time <= 0.03
    quantization_axis.plot(time[sample_window], original[sample_window], label="Original")
    quantization_axis.step(
        time[sample_window],
        quantized[sample_window],
        where="mid",
        label="4-bit quantized",
        alpha=0.8,
    )
    quantization_axis.set(
        title="Effect of 4-bit quantization",
        xlabel="Time (s)",
        ylabel="Amplitude",
    )
    quantization_axis.legend()
    quantization_axis.grid(True, alpha=0.3)

    blackout_axis.plot(time, original, label="Original", alpha=0.45)
    blackout_axis.plot(time, blackout_signal, label="Blackout signal")
    blackout_axis.plot(time, reconstructed, label="Reconstructed", linewidth=1.2)
    blackout_axis.axvspan(blackout_start, blackout_end, color="red", alpha=0.12, label="Dropout")
    blackout_axis.set_xlim(blackout_start - 0.1, blackout_end + 0.1)
    blackout_axis.set(
        title="Effect of dropout and spline reconstruction",
        xlabel="Time (s)",
        ylabel="Amplitude",
    )
    blackout_axis.legend()
    blackout_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    plt.show()


class TestBlackoutModule(unittest.TestCase):
    def setUp(self):
        """Set up a basic 440Hz sine wave for testing."""
        self.fs = 44100
        self.duration = 3.0  # 3 seconds
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        self.test_signal = np.sin(2 * np.pi * 440 * t)

    def test_simulate_quantization_reduction(self):
        """Test if quantization runs and alters the signal."""
        quantized = blackout.simulate_quantization_reduction(self.test_signal, target_bit_depth=4)
        
        # Check that the shape hasn't changed
        self.assertEqual(self.test_signal.shape, quantized.shape)
        
        # At 4-bit, the signal should be significantly degraded (not equal to original)
        self.assertFalse(np.array_equal(self.test_signal, quantized))
        
        # Check that the number of unique values is bounded by 2^4 = 16
        unique_vals = np.unique(quantized)
        self.assertLessEqual(len(unique_vals), 16)

    def test_calculate_snr(self):
        """Test SNR calculation logic."""
        # SNR of identical signals should be infinite
        snr_perfect = blackout.calculate_snr(self.test_signal, self.test_signal)
        self.assertEqual(snr_perfect, float('inf'))
        
        # SNR of degraded signal should be a finite positive number
        quantized = blackout.simulate_quantization_reduction(self.test_signal, target_bit_depth=4)
        snr_degraded = blackout.calculate_snr(self.test_signal, quantized)
        self.assertGreater(snr_degraded, 0.0)
        self.assertLess(snr_degraded, 100.0)

    def test_interpolate_dropout(self):
        """Test if the interpolation bridges the gap without changing signal length."""
        start_time = 1.0
        duration = 0.5
        
        # Test cubic spline interpolation
        reconstructed = blackout.interpolate_dropout(
            self.test_signal, 
            self.fs, 
            start_time, 
            duration, 
            method='spline'
        )
        
        # Length must remain identical
        self.assertEqual(len(self.test_signal), len(reconstructed))
        
        # The interpolated section shouldn't just be zeros
        start_idx = int(start_time * self.fs)
        end_idx = start_idx + int(duration * self.fs)
        self.assertNotEqual(np.sum(np.abs(reconstructed[start_idx:end_idx])), 0)

    def test_apply_complete_blackout_phase(self):
        """Test the master wrapper function."""
        blackout_start = 0.5
        blackout_sig, reconstructed_sig = blackout.apply_complete_blackout_phase(
            self.test_signal, 
            self.fs, 
            blackout_start
        )
        
        # Ensure lengths match
        self.assertEqual(len(self.test_signal), len(blackout_sig))
        self.assertEqual(len(self.test_signal), len(reconstructed_sig))
        
        # Verify the blackout actually contains silence where it should
        start_idx = int(blackout_start * self.fs)
        end_idx = start_idx + int(2.0 * self.fs)
        self.assertTrue(np.all(blackout_sig[start_idx:end_idx] == 0.0))
        
        # Verify the reconstructed signal is NOT silent in that same gap
        self.assertFalse(np.all(reconstructed_sig[start_idx:end_idx] == 0.0))

if __name__ == '__main__':
    show_plot = '--plot' in sys.argv
    result = unittest.main(argv=[sys.argv[0]], exit=False)
    if show_plot:
        plot_blackout_effects()
    sys.exit(not result.result.wasSuccessful())
