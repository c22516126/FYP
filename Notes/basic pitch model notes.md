## Terminology

- CQT (constant-Q transform)
    - time-frequency representation
    - similar to spectrogram but music focused
    - frequency bins line up with musical notes
    - aligns with musical scales
    - logarithmically spaced frequency bins
- frequency bin
    - a range in a spectrogram where a music note resides
    - logarithmically spaced and sized
        - aligns with semitones
        - also applies to width
        - lower tones have narrower frequency spacing
    - why is it space logarithmically
        - low notes vibrate slower - narrower frequency signature
        - high notes vibrate faster - larger harmonic spread
- harmonic stacking
    - copying CQT, shifting the position by harmonic ratios (multiplicatively)
    - shifted pitch bins are used to reinforce the same pitch bin at the same time (reinforcement done inside the CNN)
- harmonic
    - multiplication of fundemental frequency
    - different frequencies produced by a note
- pitch
    - corresponds to fundemental frequency (lowest possible frequency for a pitch)
    - accompanied by harmonics - shapes timbre
- 3D tensor
    - stack of 2d matrices - forms a 3d shape
    - depth - harmonic layers
    - shape - (harmonic layer, freq bin, time frame)
- basic pitch
    - input - 3d tensor
    - output - 3 2d posteriorgram (Yo, Yn, Yp)
- 2D posteriorgram (basic pitch output)
    - matrix containing data on musical notes, time frames, and probability of a note being active
    - shows when notes are active
    - shape
        - row - frequency bin
            - corresponds 1-1 with CQT pitch bins
        - column - time frame
        - value - probability (0-1)
- convolutional filter
- CNN (convolutional neural network)
    - output (posteriorgrams)
        - Yo - onset
            - row - processsed pitch bin (1 per semitone)
            - column - time frames
            - value - probability that a note onset occurs at this time/pitch
        - Yn - note activation
            - row - processsed pitch bin (1 per semitone)
            - column - time frames
            - value - probability note is active right now
        - Yp - pitch likelihood
            - row - processed pitch bins (3 per semitone)
            - column - time frames
            - value - pitch likelihood
- Yo - probability a note onset occurs at a given pitch and time (note start)
- Yn - probability a note is active/sustained at a given pitch and time
- Yp - probability a pitch bin is active at a given time
- post processing
    - convert posteriorgrams into:
        - discrete pitches
        - note start/end times
        - midi notes

## Research Paper Content

- problem
    - automatic music transcription (amt) is hard
    - existing models - instrument specific
    - heavy models =/= deployable (too big or slow)
    - propose lightweight, multi-instrument model
- what the approach does
    - audio -> CQT (frequency representation)
    - harmonic stacking (copies shifted to align harmonics)
    - small CNN
    - 3 outputs -> onset, note activation, multi pitch
    - post-processing - converts outputs into notes
- why this paper matters (useful for project background)
    - lightweight
    - multi instrument
    - polyphonic
    - performs well compared to big models
    - useful direction for real world AMT systems
- results
    - outperforms other lightweight instrument agnostic models
    - performs decently against heavier instrument specific systems

## Basic Pitch Structure

- 9 harmonic layers
    - original layer
    - 7 harmonic layers - shifted up
    - subharmonic layer - shifted down

## Flow

1. audio input
2. CQT -> harmonic stacking
3. CNN
    1. uses convolutional filter (learning matrix)
        1. adjusts values during training - becomes pattern detectors
        2. slide across harmonic stack cqt input
        3. produces feature maps
        4. allows output of posteriorgrams
    2. leverages harmonic shifted CQT channels
        1. align pitch harmonics onto same pitch bin (happens before CNN)
        2. reinforces fundemental frequency - provides evidence for pitch detection (happens inside CNN)
        3. uses aligned channels to estimate (after CNN and sigmoid activation) 
            1. Yo - onset probability
            2. Yn - note activity
            3. Yp - pitch likelihood
4. output
    1. Yp - pitch likelihood
    2. Yo - onset
    3. Yn - activation
5. post processing
    1. peak pick Yo - onsets
        1. pick peaks across time
        2. discard peaks with less than 0.5 probability
    2. create notes from Yo (primary pass)
        1. for each onset candidate (processed in descending time order)
        2. start a note at the onset (t₀ᵢ, fᵢ) -> onset peak for a specific pitch bin at a specific time
        3. track forwarn in Yn
        4. continue while Yn > τₙ (threshold for note activation) or it dips below the threshold for <= 11 frames
        5. end the note when it stays below the threshold for > 11 frames
    3. set newly created note's Yn to 0
    4. create fallback notes from remaining Yn
        1. find all Yn bins over the threshold
        2. for each, trace forward and backward to build notes
    5. remove notes < 120ms
    6. seperate multipitch (Yp)
        1. peak pick Yp across frequency, keep peaks over threshold
    7. construct midi notes