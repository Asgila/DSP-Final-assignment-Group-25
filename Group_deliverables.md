### Table A.1: Core requirements
Scene element - Audio effect - DSP technique required
Camera feed - Visualized ship name: 
Fouirer Transform and envelope ekstraktion. like assignment 1.

Opening transmission - Clear radio broadcast:
"Calm transmission at the beginning (clear synthesizer music, optional: space ship background sounds)" 

Instrument detection - Brief moment analyzing equipment(with acoustic feedback?):
Instrument sound with detection procedure (recorded sound, optional: device sound), like Assignment 2.1 

Anomaly interference - Narrowband hum at 6–8 kHz:
Bandpass filter of noise(6-8kHz) + dropouts 

Radiation storm - Crackle/hissing noise overlay:
The news with a Noise signal overlayed "But when passing the orbit of the Chebychev-moon, you start picking up a broadband noise that makes your signal pretty noisy".  

Moon shadow entry - Signal cuts progressively:
drop-outs while entering the moon shadow. Maybe Chebychev filter with large passband ripples.

Complete blackout - 2 seconds of silence - 

Signal return - Gradual restoration to clarity - 
step-wise return to normal. 

We will have recorded sound from us and sound from the radio.

### Table A.2 Mandatory technical components'
Filter types - At least one recursive (IIR) and one non-recursive(FIR) filter implemented

Bilinear transform - Analog-to-digital filter conversion via bilinear transform with documented pre-warping calculations (in code) 

Power spectral density (PSD) - Estimation using Welch’s method or periodogram with parameter justification

Pole-zero analysis  - z-plane pole-zero plot for at least one recursive filter with stability discussion -

Spectrogram - Visualization showing signal at relevant scene phases

Sampling frequency - Justification referencing Nyquist criterion for the chosen sampling rate.

Frequency resolution - Discussion linking window length to DFT properties and spectral resolution trade-offs -

Terminology - Proper English DSP terminology used consistently throughout documentation -

Literature citations Minimum 2 external sources cited for filter design choices and theoretical foundations