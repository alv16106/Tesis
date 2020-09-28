import librosa
import joblib
import numpy as np

def predict(songname):
    valModel = joblib.load('./models/valence.pk1')
    arModel = joblib.load('./models/arousal.pk1')
    
    header = "chroma_stft,rmse,spectral_centroid,spectral_bandwidth,rolloff,zero_crossing_rate"
    for i in range(1, 21):
        header += f' mfcc{i}'

    y, sr = librosa.load(songname, mono=True, duration=5)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    row = to_append.split()

    valence = valModel.predict(row)
    arousal = arModel.predict(row)

    print('Predicted valence: %d. Predicted arousal: %d' % (valence, arousal))