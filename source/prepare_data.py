import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn import preprocessing
import warnings
from helpers import get_class
# 44100 hrtz

warnings.filterwarnings('ignore')

def transform(directory, datasetName, getSpecto=None):
    # configure csv
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    file = open(datasetName, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # configure spectogram
    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(8,8))
    pathlib.Path('../img_data').mkdir(parents=True, exist_ok=True)
    i = 0;

    # go for it
    for filename in os.listdir('../clipswav'):
        # LOad song
        songname = f'../clipswav/{filename}'
        print('Loading', songname)
        y, sr = librosa.load(str(songname), res_type='kaiser_fast')
        print('Loaded', i)
        # Get spectogram
        if getSpecto:
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
            plt.axis('off')
            plt.savefig(f'../img_data/{filename[:-3].replace(".", "")}.png')
            plt.clf()
        # fetures
        rmse = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        i +=1;
        print(i)
        file = open(datasetName, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


def merge(targets, features):
    f = pd.read_csv(features)
    t = pd.read_csv(targets)
    t['arousal_scaled'] = (t['mean_arousal'] - 5) / 4
    t['valence_scaled'] = (t['mean_valence'] - 5) / 4
    t['emotion'] = t.apply(lambda row: get_class(row.valence_scaled, row.arousal_scaled), axis = 1)
    f['song_id'] = f['filename'].str[:-4].astype(int)
    full = t.merge(f, on='song_id', how='left')
    del full['filename']
    del full['mean_valence']
    del full['mean_arousal']
    del full['label']
    full.to_pickle('fixed')

if __name__ == "__main__":
    transform('directory', 'newData.csv')
    merge('../Annotations/static_annotations.csv', 'newData.csv')
    #data[((data['valence_scaled']>0.45) & (data['valence_scaled']<0.55)) | ((data['arousal_scaled']>0.45) & (data['arousal_scaled']<0.55))]