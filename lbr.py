import matplotlib.pyplot as plt
import numpy as np
import librosa as lbr
import soundfile as sf

wav_dir = "Benjamin_Netanyau/189.wav"
sample_rate = 16000

def wav_to_spec(path):
    sig, _= sf.read(path) #sf.read ritorna un array con i valori di ampiezza (volume) tra -1 e 1 perchè i file sono PCM encoded
    print(type(sig))
    
    print("Sample rate", sample_rate)

    stf=lbr.stft(sig) 
    spec = np.abs(stf)
    spec_db = lbr.amplitude_to_db(spec)

    print(len(spec))

    return spec_db
    

dire = "Nelson_Mandela/191.wav"
nm = wav_to_spec(dire)
plt.figure(figsize=(10,7))
img = lbr.display.specshow(nm, y_axis="log", x_axis="time", sr=sample_rate, cmap="inferno")
#plt.savefig("imgnm.png")