- pipeline overview
	1. audio
	2. preprocess
	3. model
	4. posteriorgrams
	5. note creation
	6. MIDI
	7. audio render
## Prerequisite knowledge
- audio sample
	- waveform amplitude at a specific given time
	- smallest unit
- frame
	- short chunk of audio
	- created in the model
	- represents milliseconds of sound
- CNN
	- applies filters across spatial patterns
	- uses learned convolutional filters to learn features from audio windows
	- convert learned features into pitch probability
	- outputs a vector of pitch probabilities per frame
- unwrapping
	- stitching posteriorgrams into a continuous sequence
## audio windowing
- contains fixed length of audio samples
- windows overlap 
	- notes cross boundaries - fully capture these notes
- audio windows are fed to basic pitch
- converted into learned time-pitch representation
- audio window pipeline
	1. raw audio
	2. split into overlapping windows
	3. windows go into basic pitch model
	4. model output posteriorgrams for each frame
		1. pitch
		2. onset
		3. offset
	5. stitches posteriorgrams into a single continuous timeline
	6. note creation
	7. notes to MIDI
	8. MIDI to audio
- what happens to audio windows
	- model converts windows
## posteriorgrams
- probabilities of onset/offset/activation per frame
- output of model after each audio window
- shape - time frame x 88 frequency bins
## note creation
- steps
	1. posteriorgrams used as input - per frame
	2. threshold pitch onsets and activations
	3. create notes if both thresholds are met
	4. track the note as long as the pitch is above the threshold
		1. bridge small gaps
		2. end the note if it falls under the threshold for a certain amount of frames
	5. convert frames into seconds -> make midi notes
## MIDI generation
1. take note list
2. limit polyphony
3. build a MIDI track with all the notes
4. save as a MIDI file
## audio rendering
1. load MIDI file
2. convert MIDI notes into timed synth events
	1. schedule note-on/off at each start and end time
3. render audio for each block
	1. create audio blocks
	2. trigger note on/off events at the correct times
	3. convert from stereo to mono
	4. append each audio block to a list to create the full song
4. normalize audio
	1. look for loudest sample
	2. scale the entire track by the peak audio sample
5. save as an audio file
## limitations