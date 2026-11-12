For the final sound, we will piece together snippes of sounds as specified in the core requirements. 

### Table A.1: Core requirements
**Scene element - Audio effect** - DSP technique required

**Camera feed - Visualized ship name:** 

Fouirer Transform and envelope ekstraktion. like assignment 1. The image should be mirroered. "your screen linked to the outside camera shows the mirror image of your space craft" 

**Opening transmission - Clear radio broadcast:** - Dmitrii

Use syntheziser from assignment 2 to create music "Calm transmission at the beginning (clear synthesizer music, optional: space ship background sounds)" 

**Instrument detection - Brief moment analyzing equipment(with acoustic feedback?):** - *Asger Frimor*

Instrument sound with detection procedure (recorded sound, optional: device sound), like Assignment 2.1 

**Anomaly interference - Narrowband hum at 6–8 kHz:** - *Asger Johansen*

Bandpass filter of noise(6-8kHz) + dropouts 

**Radiation storm - Crackle/hissing noise overlay:** - *Alexander*

The news with a Noise signal overlayed "But when passing the orbit of the Chebychev-moon, you start picking up a broadband noise that makes your signal pretty noisy".  

**Moon shadow entry - Signal cuts progressively:** *Alexander*

drop-outs while entering the moon shadow. Maybe Chebychev filter with large passband ripples.

**Complete blackout - 2 seconds of silence:** *Asger Frimor*

**Signal return - Gradual restoration to clarity:** - Dmitrii
step-wise return to normal. 

We will have recorded sound from us and sound from the radio.

### Table A.2 Mandatory technical components
Filter types - At least one recursive (IIR) and one non-recursive(FIR) filter implemented - Alexander, Asger Johansen

Bilinear transform - Analog-to-digital filter conversion via bilinear transform with documented pre-warping calculations (in code) - *Asger Johansen*

Power spectral density (PSD) - Estimation using Welch’s method or periodogram with parameter justification

Pole-zero analysis  - z-plane pole-zero plot for at least one recursive filter with stability discussion - *Asger Johansen, Alexander*

Spectrogram - Visualization showing signal at relevant scene phases

Sampling frequency - Justification referencing Nyquist criterion for the chosen sampling rate.

Frequency resolution - Discussion linking window length to DFT properties and spectral resolution trade-offs -

Terminology - Proper English DSP terminology used consistently throughout documentation -

Literature citations Minimum 2 external sources cited for filter design choices and theoretical foundations
