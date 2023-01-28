import librosa
import matplotlib.pyplot as plt
import librosa.display
audio_data = 'event20230121_165504_23.wav'
x , sr = librosa.load(audio_data)
print(type(x), type(sr))#<class 'numpy.ndarray'> <class 'int'>print(x.shape, sr)#(94316,) 22050



plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()