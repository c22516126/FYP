import numpy as np

# infer pitch, onset and offset of an audio window
def infer(interpreter, audioWindow):
    runner = interpreter.get_signature_runner() # create callable inferrence function

    outputs = runner(input_2=audioWindow.astype(np.float32)) # run inferrence

    # output tensors - pitch, onset, offset
    pitch = outputs["note"]      
    onset = outputs["onset"]     
    contour = outputs["contour"] 

    # remove batch windows
    pitch = np.squeeze(pitch, axis=0)
    onset = np.squeeze(onset, axis=0)
    contour = np.squeeze(contour, axis=0)

    # output posteriorgrams for each
    return pitch, onset, contour
