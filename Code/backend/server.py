import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, send_file
from flask_cors import CORS
from src.transcribe import Transcriber

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "upload")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'wav', 'mp3', "flac"}

# load model
transcriber = Transcriber()

@app.route('/', methods=['POST'])
def upload():
    if "file" not in request.files:
        return {"error": ["No file uploaded"]}, 422
    

    file = request.files['file']

    if file.filename == '':
        return {"error": "No file selected"}, 422
    
    if '.' not in file.filename:
        return {"error": "File must have an extension"}, 422

    if not file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        return{"error": ["audio format must be .mp3, .wav, or .flac"]}, 422
    

    filePath = os.path.join(UPLOAD_FOLDER, "inputFile")
    file.save(filePath)

    midiPath = transcriber.transcribe(filePath)
    return {
        "midiUrl": "/midi"
    }

@app.route('/midi')
def serve_midi():
    midiPath = os.path.join(os.path.dirname(__file__), "..", "output", "output.mid")
    return send_file(midiPath, mimetype="audio/midi")

if __name__ == "__main__":
    app.run(port=5000, debug=True)