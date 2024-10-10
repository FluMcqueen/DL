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
l = []

def spec(audio):
    return lbr.stft(audio, n_fft=frame_size, hop_length=hop)

def lib(wav):
    try:
        signal, _ = lbr.load(wav)
        sp = spec(signal)
        l.append(sp.shape)
    except :
        print(wav)

for fol in folders:
    print("-"*42)
    for i in range(100):
        if i < 10:
            path = f"{fol}/{fol}.0000{i}.wav"
        else:
            path = f"{fol}/{fol}.000{i}.wav"
        #print(path)
        lib(path)

for el in l:
    if not el == (1025, 1293):
        print(el)