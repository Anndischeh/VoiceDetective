import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from itertools import groupby


path = "[wav_file_path]"

audio, sr = librosa.load(path, sr=None)
mfccs = librosa.feature.mfcc(y=audio, sr=sr)
scaler = MinMaxScaler()
mfccs_scaled = scaler.fit_transform(mfccs.T)
kmeans = KMeans(n_clusters=2)  #n_cluster is the number of speakers
speaker_labels = kmeans.fit_predict(mfccs_scaled)

for i in range(len(speaker_labels)):
    print(f"Time Segment {i}: Speaker {speaker_labels[i]}")
    
