from pylab import *
import numpy as np
from numpy.lib import stride_tricks
import matplotlib as mpl
import scipy.io.wavfile as wav

def get_song_data(samp_freq, snd):
    channel = snd[:, 0]
    time_array = arange(0, float(snd.shape[0]), 1)

    # scale to sample duration and then to milliseconds
    time_array = time_array / samp_freq
    time_array = time_array * 1000

    return time_array, channel

def plot_time_amplitude(filename):

    samp_freq, snd = wav.read(filename)

    # dtype ranges are in16 and int32 (signed integers)    
    if snd.dtype == dtype('int16'):
        snd = snd / (2. ** 15)
    else: # 32-bit integer
        snd = snd / (2. ** 31)
    
    time_array, channel = get_song_data(samp_freq, snd)    

    plot(time_array, channel, color='k')
    ylabel('Amplitude')
    xlabel('Time (ms)')
    show()
#    savefig('time_amplitude.jpg')
#    print("Saved to time_amplitude.jpg")
    
def plot_frequency_content(filename):
    
    samp_freq, snd = wav.read(filename)

    # dtype ranges are in16 and int32 (signed integers)    
    if snd.dtype == dtype('int16'):
        snd = snd / (2. ** 15)
    else: # 32-bit integer
        snd = snd / (2. ** 31)
    
    time_array, channel = get_song_data(samp_freq, snd)    
    num_sample_points = len(channel)
    p = fft(channel) # p -> power spectrum

    nUniquePts = ceil((num_sample_points+1)/2.0)
    p = p[0:int(nUniquePts)]
    # fft return contains both magnitude and phase info and is i
    # a complex representation -- taking abs gives us magnitude info
    # of frequency components
    p = abs(p)
    p = p / float(num_sample_points)  # scale by number of points so magnitude does not
                                      #depend on length of signal or sampling frequency
    p = p**2 # square to get the power of the signal

    # odd nfft excludes Nyquist point (?? look this up)
    if num_sample_points % 2 > 0: # if we have an odd number of fft points
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2 

    freq_array = arange(0, nUniquePts, 1.0) * (samp_freq / num_sample_points)
    plot(freq_array / 1000, 10*log10(p), color='k')
    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')
    show()
#    savefig("frequency_power.jpg")
#    print("Saved to frequency_power.jpg")
    
