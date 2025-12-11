#!/usr/bin/env python3
"""
Production web UI for Kokoro TTS with chunking pipeline
"""
import os
import sys
import tempfile
import time
import io
import threading
import concurrent.futures
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np

app = Flask(__name__)

# Chunking configuration
MAX_TEXT_LENGTH = 5000
CHUNK_CHAR_LIMIT = 240
CHUNK_OVERLAP_MS = 60
CHUNK_MAX_WORKERS = 3
CHUNK_PREFETCH = 2

MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
    print(f"Error: Model files not found!")
    print(f"Make sure {MODEL_PATH} and {VOICES_PATH} are in the current directory")
    sys.exit(1)

# Initialize model once at startup
model = Kokoro(MODEL_PATH, VOICES_PATH)

def split_into_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def split_for_streaming(text, max_chars=180):
    """Finer-grained chunks for streaming to reduce per-chunk latency.
    - Start with sentence split
    - Further split long sentences by commas/semicolons/colon
    - If still long, break by words into ~max_chars segments
    """
    import re
    out = []
    for sent in split_into_sentences(text):
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        # split on commas/semicolon/colon while keeping delimiters
        parts = re.split(r'([,;:])\s*', sent)
        # rejoin pairs token+delimiter
        phrases = []
        cur = ''
        for i in range(0, len(parts), 2):
            token = parts[i].strip()
            delim = parts[i+1] if i+1 < len(parts) else ''
            chunk = (token + (delim if delim else '')).strip()
            if not chunk:
                continue
            if phrases and len(phrases[-1]) + 1 + len(chunk) <= max_chars:
                phrases[-1] = phrases[-1] + ' ' + chunk
            else:
                phrases.append(chunk)
        for ph in phrases:
            if len(ph) <= max_chars:
                out.append(ph)
            else:
                # final fallback: break by words
                words = ph.split()
                acc = []
                total = 0
                for w in words:
                    add = len(w) + (1 if total>0 else 0)
                    if total>0 and total + add > max_chars:
                        out.append(' '.join(acc))
                        acc = [w]
                        total = len(w)
                    else:
                        acc.append(w)
                        total += add
                if acc:
                    out.append(' '.join(acc))
    return out if out else [text]

def smart_chunk_text(text, max_chars=CHUNK_CHAR_LIMIT):
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        add_len = len(s) + (2 if not s.endswith(('.', '!', '?')) else 0)
        if current and cur_len + add_len > max_chars:
            chunks.append(' '.join(current))
            current = []
            cur_len = 0
        if not s.endswith(('.', '!', '?')):
            s = s + '.'
        current.append(s)
        cur_len += len(s) + 1
    if current:
        chunks.append(' '.join(current))
    return chunks if chunks else [text]

def crossfade_concat(samples_list, sample_rate, overlap_ms=CHUNK_OVERLAP_MS):
    if not samples_list:
        return np.array([], dtype=np.float32)
    if len(samples_list) == 1:
        return samples_list[0]
    overlap = max(int(sample_rate * (overlap_ms / 1000.0)), 0)
    out = samples_list[0].astype(np.float32)
    for i in range(1, len(samples_list)):
        cur = samples_list[i].astype(np.float32)
        if overlap > 0 and len(out) >= overlap and len(cur) >= overlap:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            out[-overlap:] = out[-overlap:] * fade_out + cur[:overlap] * fade_in
            out = np.concatenate([out, cur[overlap:]], axis=0)
        else:
            out = np.concatenate([out, cur], axis=0)
    np.clip(out, -1.0, 1.0, out=out)
    return out

@app.route('/synthesize-stream', methods=['POST'])
def synthesize_stream():
    """Streaming synthesis: framed raw PCM for minimal decode overhead.
    Frame per chunk: magic 'KOPC' (4 bytes) + sample_rate (u32 BE) + length (u32 BE) + float32 PCM LE bytes.
    """
    try:
        data = request.json or {}
        text = str(data.get('text', '')).strip()
        voice = str(data.get('voice', 'af_nicole'))
        language = str(data.get('language', 'en-us'))
        raw_speed = data.get('speed', 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid speed; must be a number between 0.5 and 2.0'}), 400

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({'error': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters.'}), 400

        # Smaller chunks for lower per-chunk latency
        max_chars = int((request.json or {}).get('stream_chunk_chars', 180))
        chunks = split_for_streaming(text, max_chars=max_chars)

        def generate():
            # Light pipelining: keep one chunk prefetched
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                it = iter(chunks)
                fut = None
                try:
                    first = next(it)
                except StopIteration:
                    return
                fut = pool.submit(model.create, first, voice=voice, lang=language, speed=speed)
                for nxt in it:
                    # wait current
                    try:
                        samples, sample_rate = fut.result()
                    except Exception:
                        samples, sample_rate = np.zeros((0,), dtype=np.float32), 0
                    # submit next
                    fut = pool.submit(model.create, nxt, voice=voice, lang=language, speed=speed)
                    # yield current as PCM frame
                    pcm = np.asarray(samples, dtype=np.float32).tobytes()
                    header = b'KOPC' + int(sample_rate).to_bytes(4, 'big') + len(pcm).to_bytes(4, 'big')
                    yield header + pcm
                # last outstanding
                if fut is not None:
                    try:
                        samples, sample_rate = fut.result()
                    except Exception:
                        samples, sample_rate = np.zeros((0,), dtype=np.float32), 0
                    pcm = np.asarray(samples, dtype=np.float32).tobytes()
                    header = b'KOPC' + int(sample_rate).to_bytes(4, 'big') + len(pcm).to_bytes(4, 'big')
                    yield header + pcm

        return Response(stream_with_context(generate()), mimetype='application/octet-stream', headers={'X-Content-Type-Options': 'nosniff'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        data = request.json or {}
        text = str(data.get('text', '')).strip()
        voice = str(data.get('voice', 'af_nicole'))
        language = str(data.get('language', 'en-us'))
        raw_speed = data.get('speed', 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid speed; must be a number between 0.5 and 2.0'}), 400

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({'error': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters.'}), 400

        # Chunking controls
        raw_chunk_flag = data.get('chunking', data.get('chunk', True))
        use_chunking = (
            True if isinstance(raw_chunk_flag, bool) and raw_chunk_flag else
            True if isinstance(raw_chunk_flag, (int, float)) and raw_chunk_flag != 0 else
            True if isinstance(raw_chunk_flag, str) and raw_chunk_flag.strip().lower() in ("1", "true", "yes", "on") else
            False
        )
        chunk_limit = int(data.get('chunk_chars', CHUNK_CHAR_LIMIT))
        overlap_ms = int(data.get('chunk_overlap_ms', CHUNK_OVERLAP_MS))

        if use_chunking:
            chunks = smart_chunk_text(text, max_chars=chunk_limit)
            audio_chunks = []
            sample_rate = None
            max_workers = int(data.get('chunk_workers', CHUNK_MAX_WORKERS))
            prefetch = int(data.get('chunk_prefetch', CHUNK_PREFETCH))
            max_workers = max(1, min(8, max_workers))
            prefetch = max(1, min(8, prefetch))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for idx in range(min(prefetch, len(chunks))):
                    futures.append(pool.submit(
                        model.create,
                        chunks[idx],
                        voice=voice,
                        lang=language,
                        speed=speed
                    ))
                next_idx = len(futures)
                processed = 0
                while processed < len(chunks):
                    fut = futures.pop(0)
                    try:
                        samples, sr = fut.result()
                        if sample_rate is None:
                            sample_rate = sr
                        audio_chunks.append(samples.astype(np.float32))
                    except Exception as ce:
                        # skip failed chunk
                        pass
                    processed += 1
                    if next_idx < len(chunks):
                        futures.append(pool.submit(
                            model.create,
                            chunks[next_idx],
                            voice=voice,
                            lang=language,
                            speed=speed
                        ))
                        next_idx += 1
            merged = crossfade_concat(audio_chunks, sample_rate, overlap_ms)
            samples = merged
        else:
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
