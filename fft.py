import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from pydub import AudioSegment
from numpy.lib import stride_tricks

class FFT:

    class Source:
        def __init__(self, samples, samplerate, binsize=2**10):
            self.__samples = samples
            self.__binsize = binsize
            self.__samplerate = samplerate

        @property
        def samples(self):
            return self.__samples

        @property
        def samplerate(self):
            return self.__samplerate
    
        @property
        def binsize(self):
            return self.__binsize

    class Processed:
        """ 
        timebins = sample points over time
        freqbins = logarithmically scaled frequency bin list
        decibel  = transposed and scaled amplitude matrix.
                   contains freqbins x timebins sized matrix with decibel data.
        """
        def __init__(self, spectrogram, timebins, freqbins, db_data, freq_data):
            self.__spectrogram = spectrogram
            self.__timebins = timebins
            self.__freqbins = freqbins
            self.__db_data   = db_data
            self.__freq_data = freq_data

        @property
        def spectrogram(self):
            return self.__spectrogram

        @property
        def timebins(self):
            return self.__timebins

        @property
        def freqbins(self):
            return self.__freqbins

        @property
        def db_data(self):
            return self.__db_data

        @property
        def freq_data(self):
            return self.__freq_data

    def __init__(self, source, processed):
        self.__source = source
        self.__processed = processed

    @property
    def source(self):
        return self.__source

    @property
    def processed(self):
        return self.__processed


""" short time fourier transform of audio signal """
def _stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """
def _logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    # Create an array of length freqbins, where each entry is the upper limit of the frequency band
    # e.g. [ 10hz, 100hz, 1000hz, 10khz, ... ]
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))   
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    # newspec is timebins x scale matrix i.e. with 100 samples and 10 bins it has 100 rows and 10 columns
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            # spec[:, scale[i]:scale[i+1]] is the bin-size x timebins matrix
            # where each column is a list of decibel values at the given timebin (sample) for frequencies inside the bin
            
            # 10 Hz - [ db1, db2, ... ]
            #         .................
            # 1000Hz- [ db1, db2, ... ]
            
            # so the sum of this column is the sum of the values of all frequencies in the given bin for a sample
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs


def create_fft(audiopath, binsize=2**10):
    samplerate, samples = wav.read(audiopath)
    rfft_out = _stft(samples, binsize)

    # spectrogram with frequency bins scaled logarithmically
    binned_spectrogram, freq = _logscale_spec(rfft_out, factor=1.0, sr=samplerate)

    db_matrix = 20. * np.log10(np.abs(binned_spectrogram)/10e-6)
    timebins, freqbins = np.shape(db_matrix)

    src = FFT.Source(samples, samplerate)
    processed = FFT.Processed(binned_spectrogram, timebins, freqbins, db_matrix, freq)

    return FFT(src, processed)

""" plot spectrogram"""
def plot_stft(fft, binsize=2**10, plotpath=None, colormap="jet"):
    ims = fft.processed.db_data
    timebins = fft.processed.timebins
    freqbins = fft.processed.freqbins
    freq = fft.processed.freq_data
    
    plt.figure(figsize=(25, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(fft.source.samples)/timebins)+(0.5*binsize))/fft.source.samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()
        
    plt.clf()

