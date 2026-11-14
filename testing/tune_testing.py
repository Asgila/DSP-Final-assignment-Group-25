
######################################################
# Note this testing script is made completly with AI # 
######################################################

import os
import sys
import numpy as np
import scipy.io.wavfile as wav

# Add the parent directory to the path so we can import tune.py from the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tune

def test_tuning_pipeline():
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

    print("-" * 75)
    print(f"Test Summary: {passed_tests}/{len(expected_results)} passed.")

if __name__ == "__main__":
    test_tuning_pipeline()