1. intro
	1. background
		1. explain domain
		2. define task
		3. why its hard
			1. audio signals are time domain, frequency information is not easily accessible
			2. many frequency peaks/harmonics within a pitch -> even more so with chords
		4. how its approached
		5. gap/lead into project
	2. description
		1. what system does
			1. medium (website + piano roll)
			2. characteristics (polyphony, piano focused)
		2. grid search to optimise it
		3. experiments to see cqt time/freq resolution trade-off
	3. aims and objectives
		1. transcription pipeline
		2. experiments with CQT
		3. achieve adequate accuracy using f1/precision/recall metrics
		4. optimise parameters through grid search
		5. visualisation/piano roll
		6. polyphonic input
	4. scope
		1. optimise system for piano
		2. pretrained model (basic pitch)
		3. AMT focused, no chord detection or any MIR stuff
		4. offline transcription, not online
		5. custom neural network training
	5. thesis roadmap
2. literature review
	1. intro
		1. investigate research relevant to AMT
		2. demonstrate complexity of the project
		3. approach to research
	2. alternate existing solutions
		1. basic pitch
		2. bytedance piano transcription
	3. technologies researched
		1. Flask
		2. tensorflow
		3. mir_eval
		4. librosa
		5. pretty_midi
	4. other relevant research
		1. CQT theory
		2. CNN theory
	5. existing FYPs
	6. conclusion
3. system analysis
	1. system overview
	2. requirements gathering
	3. requirements analysis
	4. logical architecture
	5. other section (make up ur own idk)
		1. data considerations
			1. midi aligned dataset
			2. piano focused
	6. initial specifications
	7. conclusions
4. system/experiment design
	1. introduction
	2. software methodology
	3. project planning
		1. schedule
	4. overview of system / experiments
		1. system
			1. transcription pipeline modularity
			2. website
		2. experiments
			1. cqt
				1. motivation
				2. params being studied
				3. hypothesis
				4. experimental design (how experiment is conducted)
					1. controlled input
					2. 1 param at a time
			2. grid search parameters
	5. fulfillment of system requirements
	6. other section if needed
		1. pipeline design
		2. post processing params
		3. grid search design
			1. param ranges
			2. search strat (1d, 2d, etc.)
		4. eval metrics
	7. conclusion
5. system/experiment development
	1. introduction
	2. software development (assuming this is also experiment/eval development?)
		1. pipeline implementation
			1. audio input
			2. cqt basic pitch implementation explanation
			3. inference
			4. stitching
			5. note creation
				1. peak pick onset matrix
				2. get pitch and interval positions of peak onsets where value is over the threshold
					1. reverse pitch and interval array - prioritize later notes for energy locking
				3. create mutable copy of frame posteriorgram - remaining energy (locking mechanism)
				4. make note array, that stores all valid onset candidates
			6. midi generation
			7. audio generation
			8. transcription function
		2. refactoring decisions
		3. cli/system integration
		4. website development/visualisation
		5. alignment
	3. conclusion
6. testing evaluation
	1. intro
	2. system testing
		1. sanity checks
		2. audio loading
		3. high level test cases
	3. system eval
		1. grid search results
		2. parameter sensitivity
		3. trade-offs
			1. precision vs recall
			2. note fragmentation vs merging (energy tolerence)
			3. effects of q factor/pitch bin changes
	4. evaluate project management
		1. eval project plan
		2. eval project execution
	5. conclusions
7. conclusion and future work
	1. intro
	2. future work
		1. editable midi after rendering
		2. chord detection
		3. beat detection
		4. key detection
		5. generalisation
		6. optimisation - make it run more efficiently
	3. conclusions
8. appendix
	1. system model and analysis
	2. design
	3. prompts with AI
	4. code samples
	5. additional appendix
---
# Diagrams to use
- transcription pipeline
	- audio input
	- windowed inference
		- audio segmentation
		- load model
		- time-frequency signal transformation
		- convolutional neural network outputs posteriorgrams
		- stitch together using center frames
	- note creation
		- threshold onsets
		- track notes until frame threshold falls below energy tolerance threshold frames
		- create offset when ^ is not true
	- convert to midi
		- convert note event attributes into midi data
- CQT
- note creation
- grid search
- system architecture
- experiments for design section
	- objective
		- params
		- to decide which params are the best, ask people 
	- subjective
		- ask people which precision vs recall results 
		- preference on recall vs precision
	- lifeheart scale