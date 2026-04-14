from src.transcribe import Transcriber

class TranscriptionCLI:
    def __init__(self, model_path):
        self.transcriber = Transcriber(model_path=model_path)

    def transcribe(self, audio_path):
        return self.transcriber.transcribe(audio_path)