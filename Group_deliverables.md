For the final sound, we will piece together snippes of sounds as specified in the core requirements. 

### Table A.1: Core requirements and Table A.5
**Scene element - Audio effect** - DSP technique required

**Camera feed - Visualized ship name:** 

Fouirer Transform and envelope ekstraktion. like assignment 1. The image should be mirroered. "your screen linked to the outside camera shows the mirror image of your space craft" 

**Opening transmission - Clear radio broadcast:** - Dmitrii

Use syntheziser from assignment 2 to create music "Calm transmission at the beginning (clear synthesizer music, optional: space ship background sounds)" 

**Instrument detection - Brief moment analyzing equipment(with acoustic feedback?):** - *Asger Frimor*

Instrument sound with detection procedure (recorded sound, optional: device sound), like Assignment 2.1 
Harmonic analysis + classification demo

**Anomaly interference - Narrowband hum at 6–8 kHz:** - *Asger Johansen*

Bandpass filter of noise(6-8kHz) + dropouts, FIR band-stop filter design and IIR band-stop via bilinear transform + pre-warping. 

**Radiation storm - Crackle/hissing noise overlay:** - *Alexander*

The news with a Noise signal overlayed "But when passing the orbit of the Chebychev-moon, you start picking up a broadband noise that makes your signal pretty noisy".  
Broadband noise genera- tion + spectral shaping, and PSD estimation using Welch’s method or periodogram

**Moon shadow entry - Signal cuts progressively:** *Alexander*

drop-outs while entering the moon shadow. Maybe Chebychev filter with large passband ripples.
Windowing functions for temporal masking, Pole-zero plot for recursive filter stability.

**Complete blackout - 2 seconds of silence:** *Asger Frimor*
Signal interpolation during dropout, Quantization depth reduction simulation

**Signal return - Gradual restoration to clarity:** - Dmitrii

step-wise return to normal. 
Cross-fade transitions be- tween scenes.  Resampling effects after bandwidth reduction


We will have recorded sound from us and sound from the radio.

**Final Soundtrack ALL parts prior to this should be finished at the end of week 11**
**Documentation. We are aiming to write the documentation in week 12**

### Table A.2 Mandatory technical components
Filter types - At least one recursive (IIR) and one non-recursive(FIR) filter implemented - Alexander, Asger Johansen, Dmitrii

Bilinear transform - Analog-to-digital filter conversion via bilinear transform with documented pre-warping calculations (in code) - *Asger Johansen*

Power spectral density (PSD) - Estimation using Welch’s method or periodogram with parameter justification - Alexander

Pole-zero analysis  - z-plane pole-zero plot for at least one recursive filter with stability discussion - *Asger Johansen, Alexander*

Spectrogram - Visualization showing signal at relevant scene phases

Sampling frequency - Justification referencing Nyquist criterion for the chosen sampling rate. - Dmitrii

Frequency resolution - Discussion linking window length to DFT properties and spectral resolution trade-offs -

Terminology - Proper English DSP terminology used consistently throughout documentation -

Literature citations Minimum 2 external sources cited for filter design choices and theoretical foundations
