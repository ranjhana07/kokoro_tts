#!/usr/bin/env python3
"""
Simple web UI for Kokoro TTS
"""
import os
import sys
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np

app = Flask(__name__)

# Initialize Kokoro model
MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
    print(f"Error: Model files not found!")
    print(f"Make sure {MODEL_PATH} and {VOICES_PATH} are in the current directory")
    sys.exit(1)

# Available voices
VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
]

# Available languages (based on Kokoro TTS actual support)
LANGUAGES = [
    "en-us",    # English (US)
    "en-gb",    # English (UK)
    "fr-fr",    # French
    "it",       # Italian
    "ja",       # Japanese
    "cmn"       # Chinese (Mandarin)
]

@app.route('/')
def index():
    return render_template('index.html', voices=VOICES, languages=LANGUAGES)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'af_nicole')
        language = data.get('language', 'en-us')
        speed = float(data.get('speed', 1.0))
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Initialize model
        model = Kokoro(MODEL_PATH, VOICES_PATH)
        
        # Generate audio
        samples, sample_rate = model.create(
            text,
            voice=voice,
            lang=language,
            speed=speed
        )
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, samples, sample_rate)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='output.wav'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Kokoro TTS Web UI...")
    print("Open your browser at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
