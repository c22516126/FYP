
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
## Report Structure
1. CQT Theory
	1. intro to time-freq analysis (context)
		1. why spectral representations are needed
		2. stft
	2. limitations of stft
		1. linear frequency spacing
		2. fixed window size
		3. poor low freq resolution
	3. CQT overview (original paper)
		1. constant q ratio
		2. logarithmic freq spacing
		3. musical alignment
	4. variable window lengths + implications
		1. window length = Q * sample rate / frequency center
		2. low frequency -> long windows
		3. high frequency -> high windows
		4. time vs freq tradeoff
		5. onset smearing
	5. advantages and limitations of CQT
		1. advantages
			1. musical pitch alignment
			2. better low freq resolution
		2. limitations
			1. temporal smearing at low frequencies
			2. frame based timing contraints
			3. freq dependant resolution
	6. modern CQT implementaions (CQT toolbox paper)
		1. multi sample rate
		2. downsampling across octaves
		3. computational efficiency
	7. summary
		1. tradeoffs
		2. why CQT is suitable
2. CQT implementation in basic pitch
	1. role of CQT in pipeline
		1. audio -> CQT -> CNN -> MIDI
	2. implementation approach
		1. multi rate CQT (schorkhuber based)
		2. kernel reuse across octaves
		3. downsample strat
	3. key params and effects
		1. hop length
			1. time resolution
			2. frame spacing
		2. bins per octave
			1. freq resolution
			2. pitch granularity
		3. min frequency
			1. coverage of pitch range
		4. window function
			1. hann window
			2. effect on spectral leakage
		5. q factor
			1. controls window length
			2. tradeoff tuning
	4. analysis window length in practice
		1. why low notes behave different
		2. effect on onset detection
	5. implications for CNN input representation
		1. time freq structure
		2. frame alignment
		3. pitch dependant resolution
	6. limitations of implementation
		1. downsampling artifacts
		2. temporal resolution issues
		3. approximation vs true CQT
	7. summary
3. CQT Experiments
	1. experimental setup
	2. effectt of hop length on time resolution
	3. effect of freq resolution (bins)
	4. effect of q factor
	5. analysis of observed tradeoffs
		1. what changed
		2. why it changed
		3. link back to theory
	6. limitations and observed fail cases
		1. low freq onset issues
		2. fast note transitions
		3. noise sensitivity
	7. summary of experimental findings
- diagrams
	- sec 1
		- stft vs CQT
		- variable window
		- pitch bin mapping (frequency center calculation)
	- sec 2
		- pipeline flow
		- kernel matching