# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile
#
# sample_rate, samples = wavfile.read('03.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
# print (max(spectrogram), min(spectrogram))
# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()


# import the pyplot and wavfile modules

import matplotlib.pyplot as plot
import os
from scipy.io import wavfile
path = "//"
path = "E:/sdr/meteors/01/"

l = os.listdir(path)
l.sort()
print (l)
# Read the wav file (mono)
#l = ['event20230102_141056_26.wav']
for w in l:
    try:
        if w.endswith('.wav'):
            samplingFrequency, signalData = wavfile.read(path + w)
            #samplingFrequency, signalData = wavfile.read( w)

            # Plot the signal read from wav file

            plot.subplot(211)

            plot.title(w)

            plot.plot(signalData)

            plot.xlabel('Sample')

            plot.ylabel('Amplitude')

            plot.subplot(212)

            plot.specgram(signalData, Fs=samplingFrequency, cmap='jet')
            plot.colorbar(orientation="horizontal", pad=0.3)

            plot.xlabel('Time')

            plot.ylabel('Frequency')

            plot.show()
    except ValueError:
        pass