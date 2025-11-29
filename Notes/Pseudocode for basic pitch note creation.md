 for each pitch p:
    find onset_frames = frames where onset_probs > onset_threshold

    for each onset_frame t0:
        t = t0
        while pitch_probs[t,p] > activation_threshold:
            if offset_probs[t,p] > offset_threshold:
                break
            t += 1
        
        end_time = t
        save note (pitch=p, start=t0, end=end_time)
