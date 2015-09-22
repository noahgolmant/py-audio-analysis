from whole_fft import plot_time_amplitude, plot_frequency_content
from short_fft import plot_stft
import matplotlib as mpl
from pydub import AudioSegment

""" convert mp3 to wav """
def get_wav(filename):
    extension = filename.split(".")[-1]
    if extension == "wav":
        return filename
    elif extension == "mp3":
        sound = AudioSegment.from_mp3(filename)
        newfilename = filename.replace(extension, "wav")
        sound.export(newfilename, format="wav")
        print("Converted {0} to {1}".format(filename, newfilename))
        return newfilename

def choose_fft(filename, choice):
    filename = get_wav(filename)
    
    if choice == '0':
        plot_time_amplitude(filename)
    elif choice == '1':
        plot_frequency_content(filename)
    elif choice == '2':
        plot_stft(filename)

def main():
    print("---------------------------")
    filename = input("Enter filename: ")
    print("---------------------------")
    print("Amplitude vs Time       [0]")
    print("Power (dB) vs Frequency [1]")
    print("Short time FFT specgram [2]")
    print("---------------------------")
    choice = input("Select a graph number: ")
    
    choose_fft(filename, choice)

if __name__ == '__main__':
    mpl.rcParams['agg.path.chunksize'] = 100000 
    main()
