
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
    unittest.main()