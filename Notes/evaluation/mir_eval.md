### notes for tmrw
- figure out what's bipartite matching
- read entirety of transcription
- breakdown how it works
- attempt to run it myself
---
- each task has a submodule
- each metric is a function within a submodule
---
# Relevant MIR Tasks
## [transcription](https://mir-eval.readthedocs.io/latest/api/transcription.html)
### what does it do
- count how many notes match reference and how many don't
- based on counts, estimate:
	- f-measure
	- precision
	- recall
	- overlap ratio
- ground truth
	- setup 1
		- onset within 50ms of reference
		- f0 within quarter tone
	- setup 2
		- same as setup 1 but also has either (larger picked):
			- offset within 50ms
			- offset value within 20% of reference duration
- how is it done
	- bipartite graph matching to find optimal pairing of notes
### metrics 
- mir_eval.transcription.precision_recall_f1_overlap()
	- precision, recall, F-measure, average overlap ratio
	- note is correct if:
		- pitch and onset are close to reference note
		- offset is optional
- mir_eval.transcription.onset_precision_recall_f1()
	- precision, recall, F-measure
		- note correct if:
			- onsets are sufficiently close
			- only onsets taken into account - pitch values can be different
- mir_eval.transcription.offset_precision_recall_f1()
	- precision recall, F-measure
		- correct if:
			- offset sufficiently close (read ground truth setup 2^)
### Parameter shapes
- intervals
	- 2d arrays of [onset, offset], where each row is a seperate note interval
- pitch
	- mir_eval uses hz for pitch
	- use mir_eval.util.midi_to_hz to convert midi to hz
### Functions
- validation
	- mir_eval.transcription.validate(ref_intervals, ref pitches, est_intervals, est pitches)
		- purpose: 
			- check if input for metric look like time intervals and pitch list
			- throw errors if not
		- parameters:
			- ref_intervals : np.ndarray, shape=(n,2) -> 2d array
				- reference note time intervals (onset and offset)
			- ref_pitches : np_ndarray, shape=(n,) -> 1d array
				- reference pitch values in hertz
			- est_intervals : np.ndarray, shape=(m,2) -> 2d array
				- estimated note intervals
			- est_pitches : np_ndarray, shape=(m,) -> 1d array
				- estimated pitch values
	- mir_eval.transcription.validate_intervals(ref_intervals, est_intervals)
		- purpose: 
			- check if input look like time intervals
			- throw error if not
		- parameters:
			- intervals, same as above
# References
[mir_eval research paper](https://colinraffel.com/publications/ismir2014mir_eval.pdf)
[MIREX note tracking subtask](http://www.music-ir.org/mirex/wiki/2015:Multiple_Fundamental_Frequency_Estimation_%26_Tracking_Results_-_MIREX_Dataset#Task_2:Note_Tracking_.28NT.29)
