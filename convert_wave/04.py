import matplotlib.pyplot as plt
import scipy.io.wavfile
sample_rate, X = scipy.io.wavfile.read('event20230103_083707_47.wav')
print(sample_rate, X.shape)
plt.xlabel('Время, c', fontsize='x-large')
plt.ylabel('Частота, Гц', fontsize='x-large')
plt.title('Спектрограмма для колбы')
cmap = plt.get_cmap('magma')
plt.specgram(X, NFFT=256, pad_to=256, mode='magnitude', Fs=sample_rate, cmap=cmap)
plt.colorbar()
plt.show()