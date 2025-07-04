from flask import Flask, request, render_template, jsonify
import os
import librosa
import librosa.display
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from IPython.display import Audio
import matplotlib.pyplot as plt
import uuid
import atexit

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

encoder = joblib.load('speech_rec_encoder.pkl')
model = load_model('speech_rec.h5')

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc



def waveplot_func(data, sr):
    plt.figure(figsize=(10,4))
    plt.title("Waveplot", size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    
def spectogram(data, sr):
     x = librosa.stft(data)
     xdb = librosa.amplitude_to_db(abs(x))
     plt.figure(figsize=(11,4))
     plt.title("Spectogram", size=20)
     librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
     plt.colorbar()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
     if 'audio' not in request.files:
          return jsonify({'error': 'No file uploaded'})

     file = request.files['audio']
     if file.filename == '':
          return jsonify({'error': 'Empty filename'})

     filename = secure_filename(file.filename)
     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
     file.save(filepath)

     features = extract_mfcc(filepath).reshape(1, -1)
     pred = model.predict(features)
     predicted_label = encoder.inverse_transform(pred)[0][0]
     # print(predicted_label)
     os.remove(filepath)

     return jsonify({'emotion': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
