import numpy as np
import librosa as lbr
import librosa.display as display
import wave
import math
import matplotlib.pyplot as plt


folders = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

hop = 512
frame_size = 2048
sr = 22050
l_a = []

def spec(audio):
    return lbr.stft(audio, n_fft=frame_size, hop_length=hop)

def lib(wav, lista):
    try:
        signal, _ = lbr.load(wav)
        sp = spec(signal)
        lista.append(sp)
    except wave.Error as e:
        print(wav)

for fol in folders[:4]:
    for i in range(100):
        if i < 10:
            path = f"{fol}/{fol}.0000{i}.wav"
        else:
            path = f"{fol}/{fol}.000{i}.wav"
        #print(path)
        lib(path, l_a)

print("-"*42)

l_b = []

for fol in folders[4:7]:
    for i in range(100):
        if i < 10:
            path = f"{fol}/{fol}.0000{i}.wav"
        else:
            path = f"{fol}/{fol}.000{i}.wav"
        #print(path)
        lib(path, l_b)