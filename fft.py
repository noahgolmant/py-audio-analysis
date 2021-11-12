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


def _stft(signal, frame_size, overlap_fac=0.5, window=np.hanning):
    """ short-time fourier transform of an audio signal
    -------------
    DESCRIPTION:
    Output is a matrix where each row represents a frame of time length frame_size
    and each column entry is the loudness of the frequency specified by the column index.
    ------------
    INPUT:
    signal: amplitude v time raw data of the audio stream
    frame_size: size in samples of each frame to which it applies the FFT
    overlap_fac: overlap between frames when applying windowing function
    window: windowing function to reduce signal leakage in source audio stream
    """
    win = window(frame_size)
    # number of samples to shift by for each application of the windowing function
    hop_size = int(frame_size - np.floor(overlap_fac * frame_size))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size/2.0))), signal)    
    # columns in signal v time matrix that we apply the windowing to
    cols = np.ceil( (len(samples) - frame_size) / float(hop_size)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))
    # apply byte-ordered indexing scheme to sample array
    # creates cols x frame_size frames of samples
    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size), strides=(samples.strides[0]*hop_size, samples.strides[0])).copy()
    frames *= win
    # apply real FFT to each frame
    return np.fft.rfft(frames)    
    
def _logscale_spec(fft_out, sr=44100, factor=20.):
    """ scale frequency axis logarithmically
    ------------
    DESCRIPTION:
    Scale the fft output logarithmically.
    Scales length of original FFT output so that each column entry
    represents loudness at frequency *bin* at the column index.
    Each bin is frequency band with logarithmic scaling ( 10 Hz, 100 Hz, ...)

    Returns scaled fft output and an array containing the center of each frequency band.
    -----------
    INPUT:
    fft_out: output of stft applied to each frame of the original signal
    sr: samplerate of audio signal
    factor: logarithmic scaling factor
    """
    timebins, freqbins = np.shape(fft_out)

    # Create an array of length freqbins, where each entry is the upper limit of the frequency band
    # e.g. [ 10hz, 100hz, 1000hz, 10khz, ... ]
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))   
    # create spectrogram with new freq bins
    scaled_fft = np.complex128(np.zeros([timebins, len(scale)]))
    # scaled_fft is a matrix of sample bins x frequency bins
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            # spec[:, scale[i]:scale[i+1]] is the bin-size x timebins matrix
            # where each column is a list of decibel values at the given frame for frequencies inside the bin
            
            # 10 Hz - [ db1, db2, ... ]
            #         .................
            # 1000Hz- [ db1, db2, ... ]
            
            # so the sum of this column is the sum of the values of all frequencies in the given bin for a sample
            scaled_fft[:,i] = np.sum(fft_out[:,scale[i]:], axis=1)
        else:        
            scaled_fft[:,i] = np.sum(fft_out[:,scale[i]:scale[i+1]], axis=1)
  
    # allfreqs = all frequencies represented by fft  
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqbin_centers = []
    # get array of frequencies at center of each bin
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqbin_centers += [np.mean(allfreqs[scale[i]:])]
        else:
            freqbin_centers += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return scaled_fft, freqbin_centers


def create_fft(audiopath, binsize=2**10):
    samplerate, samples = wav.read(audiopath)
    # apply real FFT
    rfft_out = _stft(samples, binsize)

    # spectrogram with frequency bins scaled logarithmically
    scaled_fft, freqbin_centers = _logscale_spec(rfft_out, factor=1.0, sr=samplerate)

    db_matrix = 20. * np.log10(np.abs(scaled_fft)/10e-6)
    timebins, freqbins = np.shape(db_matrix)

    src = FFT.Source(samples, samplerate)
    processed = FFT.Processed(scaled_fft, timebins, freqbins, db_matrix, freqbin_centers)

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

