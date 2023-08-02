import sounddevice as sd
import numpy as np
import librosa
import math
from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='/static')

app = Flask(__name__)


def calculate_bpm(y, sr):
    y_mono = librosa.to_mono(y)
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return tempo


def record_audio(duration, sample_rate):
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate, channels=2)
    sd.wait()  # Wait until the recording is finished
    return audio_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate_bpm', methods=['POST'])
def calculate_bpm_route():
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sample rate in Hz (standard for most audio)

    try:
        audio_data = record_audio(duration, sample_rate)
        # Transpose audio_data from shape (duration_samples, channels) to (channels, duration_samples)
        audio_data = np.transpose(audio_data)

        # Check if the audio contains any significant amplitude
        max_amplitude = np.max(np.abs(audio_data))
        silence_threshold = 0.1  # Adjust this threshold as needed
        if max_amplitude < silence_threshold:
            return "Error: No significant audio detected", 400

        # Calculate BPM
        bpm = math.ceil(calculate_bpm(audio_data, sample_rate))
        return f"{bpm} bpm"
    except Exception as e:
        return "Error: Track not found or recording error", 400  # Return an error response with status code 400

if __name__ == "__main__":
    app.run(debug=True)
