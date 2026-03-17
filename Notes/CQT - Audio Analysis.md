
| Step                | File               |
| ------------------- | ------------------ |
| Audio loading       | `inference.py`     |
| Windowing           | `inference.py`     |
| CQT computation     | `nnaudio.py`       |
| CQT call            | `models.py`        |
| Harmonic stacking   | `nn.py`            |
| CNN model           | `models.py`        |
| Inference stitching | `inference.py`     |
| Note creation       | `note_creation.py` |
| MIDI export         | `inference.py`     |
- report sections
	- audio input/preprocessing
		- load audio with librosa
		- resample to 22050
		- covert audio to mono
		- divide waveform into 2 sec segments (audio window length)
			- each segment is model input tensor
			- 30 frame overlap
	- segmentation/audio windowing
		- 2 secs per second
		- 22050 samples per sec
		- audio segments individually processed
	- time freq representation (CQT)
		- 
	- spectral normalization/scaling/compression
		- 
	- harmonic feature augmentation (harmonic stacking)
		- 
	- final representation

---
## Default Constants
- min freq - 32.7
- number of freq bins - 84
- bins per octave - 12
- cqt window - hann
## Low pass filer
- remove high frequencies
	- causes aliasing
## Early downsample
- optimization step
- reduce computation when analyzing low frequencies
- low frequencies have large windows
- less samples require shorter windows
## Window creation
- function
	- get_window_dispatch()
- creates analysis window
	- hann window
- weighting function applied to signal segment
- able to change window types for experiments
	- hann
	- hamming
	- blackman
## CQT kernel creation
- function
	- create_cqt_kernels()
	- build filter
- analysis window length formula
	- fftLen = 2 **  next_power_of_2(np.ceil(q * fs / fmin))
- window length for each pitch bin
	- length = ceil(q * fs / freq)
- meaning in english
	- window size = number of cycles * samples per cycle
- kernel creation
	- window * complex sinusoid
	- gives localized sinusoid
	- kernel detects energy at specific musical pitch
## CQT computation (actual transform)
- function
	- get_cqt_complex()
- operation
	- tf.nn.conv1d()
	- kernel slides across audio
	- audio * pitch detector
		- repeats across time
- output
	- cqt_real
	- cqt_imag
## Downsample helper
- downsampling_by_n()
	- lowpass -> stride convolution -> downsample
## Padding layers
- classes
	- reflectionPad1D
	- constantPad1D
- purpose
	- padding required when kernel slides over edges
## Build function
1. compute Q factor
	1. Q = factor_scale / (2^(1/bins_per_octave) - 1)
		1. controls freq vs time resolution
2. calculate number of octaves
	1. n_octaves = ceil(n_bins / bins_per_octave)
	2. example
		1. 84 bins
		2. 12 per octave
		3. 7 octaves
3. determine top octave freq
	1. compute highest octave first
	2. use multi resolution cqt instead of computing all bins at once
4. create cqt kernels
	1. pitch filters
5. convert kernels to tensorFlow
## Forward pass
1. reshape input
2. early down sample
3. compute top octave cqt
4. compute lower octaves
5. trim bins
	1. removes extras
6. normalization
	1. match output to librosa CQT scale
7. magnitude conversion
	1. sqrt(real^2 +image^2)
## Alias

---
# Notes
![[Pasted image 20260316180919.png]]
- frequency
	- 
- window
	- sample rate * Q / freq center
- bin bandwidth
	- 
- q 
- cycles in window