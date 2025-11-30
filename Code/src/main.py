# main.py
import argparse
from CLI.CLI import TranscriptionCLI
from config import MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Audio-to-MIDI CLI")
    parser.add_argument("audio", help="Relative path to input audio")
    args = parser.parse_args()

    cli = TranscriptionCLI(
        model_path=MODEL_PATH
    )
    cli.transcribe(args.audio)

if __name__ == "__main__":
    main()
