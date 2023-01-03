




# Indicator signal reference implementation v1.0, (c) Tom Roelandts, 2014-10-25.
# Permission is hereby granted to use this code for academic purposes.
#
# For more information, see
# http://tomroelandts.com/articles/meteor-detection-for-brams-using-only-the-
#                                  time-signal
#
# This script was developed and tested using the Anaconda Python distribution.
#

from os.path import basename
import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib.pylab import specgram
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
# Function definitions.
#

def find_carrier(sig, Fs, Fmin, Fmax):
    # Determines the location of the carrier.
    #
    # INPUT
    #   sig:       signal
    #   Fs [Hz]:   sample rate
    #   Fmin [Hz]: minimum frequency of the carrier
    #   Fmax [Hz]: maximum frequency of the carrier
    #
    # OUTPUT
    #   Returns the frequency of the carrier in Hz.

    # Pad signal length to power of two to speed up fft.
    print (2.0 ** np.ceil(np.log2(sig.size)))
    padd_sig = np.zeros(int(2.0 ** np.ceil(np.log2(sig.size))))
    padd_sig[0: sig.size] = sig
    ps = 20. * np.log10(np.abs(np.fft.rfft(padd_sig)))  # Power spectrum.
    freqs = np.fft.rfftfreq(padd_sig.size, d=1./Fs)
    FminIdx = np.round(Fmin / (Fs / 2) * ps.size).astype(int)
    FmaxIdx = np.round(Fmax / (Fs / 2) * ps.size).astype(int)
    Fc = freqs[np.argmax(ps[FminIdx : FmaxIdx + 1]) + FminIdx]
    return Fc, freqs, ps

def band_pass_filter(sig, Fs, Fc):
    # Applies a band-pass filter to the signal.
    #
    # INPUT
    #   sig:     signal
    #   Fs [Hz]: sample rate
    #   Fc [Hz]: carrier frequency
    #
    # OUTPUT
    #   Returns the band-pass-filtered signal.
    #
    # Remark: a straightforward optimization is combining both filters by
    # convolving their impulse responses. This has not been done  here for
    # clarity. For more information, see
    # http://tomroelandts.com/articles/how-to-create-simple-band-pass-and-band-
    #                                  reject-filters

    # Low-pass filter settings.
    Fcut = (Fc + 30.) / Fs      # Cutoff frequency.
    b = 0.001                   # Roll-off.
    N = int(np.ceil((4. / b)))  # Length of filter.
    if not N % 2: N += 1        # Make sure that N is odd.
    n = np.arange(N)
    # Compute sinc filter.
    h = np.sinc(2 * Fcut * (n - (N - 1) / 2.))
    # Apply Blackman window.
    h *= np.blackman(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Apply low-pass filter.
    sig = fftconvolve(sig, h, mode='valid')

    # High-pass filter settings.
    Fcut = (Fc - 30.) / Fs      # Cutoff frequency.
    b = 0.001                   # Roll-off.
    N = int(np.ceil((4. / b)))  # Length of filter.
    if not N % 2: N += 1        # Make sure that N is odd.
    n = np.arange(N)
    # Compute sinc filter.
    h = np.sinc(2 * Fcut * (n - (N - 1) / 2.))
    # Apply Blackman window.
    h *= np.blackman(N)
    # Normalize to get unity gain.
    h /= np.sum(h)
    # Convert to high-pass.
    h = -h
    h[N // 2] += 1
    # Apply high-pass filter.
    sig = fftconvolve(sig, h, mode='valid')

    return sig

def indicator_signal(sig, len_long, len_short):
    # Computes the indicator signal.
    #
    # INPUT
    #   sig:       signal
    #   len_long:  length of long running average; must be odd
    #   len_short: length of short running avarage; Must be odd
    #
    # OUTPUT
    #   Returns the indicator signal.

    # Compute running averages.
    sig_pow = sig ** 2
    ra_long = fftconvolve(sig_pow, np.ones(len_long) / len_long, \
                          mode='valid')
    ra_short = fftconvolve(sig_pow, np.ones(len_short) / len_short, \
                           mode='valid')

    # Correct for the different delay of the two filters.
    len_diff = len_long - len_short
    ra_short = ra_short[len_diff // 2 : -(len_diff // 2)]

    # Compute indicator signal.
    ra_long[ra_long < 1e-5] = 1e-5  # "Heuristic optimization"
    ind_sig = ra_short * len_short / (ra_long * len_long)
    return ind_sig

def detect_meteors(ind_sig, t, min_between):
    # Detects meteor using the indicator signal.
    #
    # INPUT
    #   ind_sig:     indicator signal
    #   t:           threshold
    #   min_between: minimum number of samples between meteors
    #
    # OUTPUT
    #   Returns the start indices of the meteors that were detected.

    during = False                    # True during a meteor.
    meteors = np.zeros(0, dtype=int)  # Start indices of meteors.
    end_prev = 0                      # End of previous meteor.
    for i in range(ind_sig.size):
        if not during:
            if ind_sig[i] >= t:
                during = True
                if i - end_prev >= min_between:
                    meteors = np.append(meteors, i)
        else:
            if ind_sig[i] < t:
                during = False
                end_prev = i
    return meteors

def create_spectrogram(sig, Fs, Fc, meteors, rsam):
    # Creates a spectrogram.
    #
    # INPUT
    #   sig:     signal
    #   Fs [Hz]: sample rate
    #   Fc [Hz]: carrier frequency
    #   meteors: start indices of meteors
    #   rsam:    number of samples that was removed

    nfft = 16384
    Pxx, freqs, bins, _ = specgram(sig, NFFT=nfft, Fs=Fs, \
                                   noverlap=round(nfft * 0.9))
    freqs[freqs < Fc - 100] = 0
    freqs[freqs > Fc + 100] = 0
    selection = np.nonzero(freqs)[0]
    Pxx = np.flipud(Pxx[selection, :])

    fig = plt.figure()
    ax = plt.gca()
    ax.plot((meteors + rsam // 2) / float(sig.size) * Pxx.shape[1], \
            (Pxx.shape[0] - 100) * np.ones(meteors.size), 'ro', markersize=3)
    im = ax.imshow(10 * np.log10(Pxx / np.max(Pxx)), interpolation='none')
    ax.axis('off')
    clim = im.get_clim()
    im.set_clim(10 * np.log10(np.mean(Pxx) / np.max(Pxx)) - 5, clim[1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)

#
# Main code.
#

def main():
    # Configuration.
    outdir = 'output'
    path = 'event20230103_083707_47.wav'
    #path = 'wavs/RAD_BEDOUR_20140502_1440_BEUCCL_SYS001.wav';
    #path = 'wavs/RAD_BEDOUR_20131214_0630_BEUCCL_SYS001.wav';
    #path = 'wavs/RAD_BEDOUR_20130326_1425_BEOTTI_SYS001.wav';
    Fmin = 800.   # Begin of search range for carrier frequency[Hz].
    Fmax = 1200.  # End of search range for carrier frequency [Hz].
    t = 0.025     # Threshold for detection.
    len_long = 30001  # [samples] Length of long running average. Must be odd.
    len_short = 101   # [samples] Length of short running avarage. Must be odd.
    min_between = 5512  # [samples] Minimum number of samples between meteors.
    verbose = True

    base = basename(path)
    prefix = outdir + '/' + base[:base.rfind('.')] + '_'

    # Load wav file.
    Fs, sig = read(path)  # Fs: sample rate.
    sig = sig.astype(float)
    sig /= np.max(np.abs(sig))  # Normalize.
    orig_sig = sig

    # Find the location of the carrier.
    Fc, freqs, ps = find_carrier(sig, Fs, Fmin, Fmax)

    # Apply a band-pass filter to the signal to reduce noise.
    fsig = band_pass_filter(sig, Fs, Fc)

    # Compute the indicator signal.
    ind_sig = indicator_signal(fsig, len_long, len_short)

    # Compensate the filtered signal for the delay of the indicator signal.
    fsig = fsig[len_long // 2 : -(len_long // 2)]

    # Due to the filtering, a certain number of samples at the beginning and the
    # end of the wav file are not used. This can be avoided by using part of the
    # previous and following wav files. If this is not done, then counts have to
    # be corrected by dividing by used_frac.
    rsam = orig_sig.size - ind_sig.size              # Removed samples.
    used_frac = float(ind_sig.size) / orig_sig.size  # Fraction of samples used.

    # Detect meteors.
    meteors = detect_meteors(ind_sig, t, min_between)

    if verbose:
        print ('meteors detected: ' + str(meteors.size))
        print ('removed samples: ' + str(rsam) + '(' + \
              str(np.round(used_frac * 100, decimals=2)) + '% of samples used)')
        print ('carrier frequency: ' + str(int(round(Fc))) + ' Hz')

        plt.figure()
        plt.plot(sig, '.', markersize=1)
        plt.savefig('Signal.png')

        plt.figure()
        plt.plot((sig ** 2), '.', markersize=1)
        plt.savefig('SignalPower.png')

        plt.figure()
        plt.plot(freqs, ps)
        plt.savefig('Spectrum.png')

        plt.figure()
        plt.plot(fsig, '.', markersize=1)
        plt.savefig('FilteredSignal.png')

        plt.figure()
        plt.plot(fsig ** 2, '.', markersize=1)
        plt.savefig('FilteredSignalPower.png')

        plt.figure()
        plt.plot(ind_sig, '.', markersize=1)
        plt.plot(np.ones(ind_sig.size) * t, 'r.', markersize=1)
        plt.savefig('IndicatorSignal.png')

        plt.figure()
        plt.plot(fsig ** 2, '.', markersize=1)
        plt.plot(meteors, np.zeros(meteors.size), 'ro', markersize=5)
        plt.savefig('FilteredSignalPowerWithDetections.png')

        create_spectrogram(orig_sig, Fs, Fc, meteors, rsam)
        plt.savefig('SpectrogramWithDetections.png', dpi=200, \
                    bbox_inches='tight')

main()