'''
Audio par
- num canali (1 o 2)
- sample width (num bytes)
- framerate (num samp al secondo)
- tot frames
- values of a frame (binary)
'''

import numpy as np
import librosa as lbr
import librosa.display
import wave
import matplotlib.pyplot as plt


classic_wav = "classical/classical.00000.wav"
metal_wav = "metal/metal.00000.wav"

def par(wav):
    obj = wave.open(wav, 'rb')
    print("Parmeters", obj.getparams())

    print(obj.getparams())
    obj.close()

par(classic_wav)
print("-"*42)
par(metal_wav)
print("-"*42)

sr = 22050
classical, _ = lbr.load(classic_wav)
metal, _ = lbr.load(metal_wav)

print(f"Sample duration {1/sr:.6f} seconds")

plt.figure(figsize=(15,17))

plt.subplot(2,1,1)
librosa.display.waveshow(classical)
plt.title("Class")

plt.subplot(2,1,2)
librosa.display.waveshow(metal)
plt.title("Metal")

plt.show()

# plt.savefig("tutwav.png")

FRAME_SIZE = 1024
HOP_LENGTH = 512

def amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

ae_classical = amplitude_envelope(classical, FRAME_SIZE, HOP_LENGTH)
ae_metal = amplitude_envelope(metal, FRAME_SIZE, HOP_LENGTH)

frames = range(len(ae_classical))
tcl = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

framesm = range(len(ae_metal))
tm = librosa.frames_to_time(framesm, hop_length=HOP_LENGTH)

plt.figure(figsize=(15, 17))

ax = plt.subplot(2, 1, 1)
librosa.display.waveshow(classical, alpha=0.5)
plt.plot(tcl, ae_classical, color="r")
plt.ylim((-1, 1))
plt.title("Classical")

plt.subplot(2, 1, 2)
librosa.display.waveshow(metal, alpha=0.5)
plt.plot(tm, ae_metal, color="r")
plt.ylim((-1, 1))
plt.title("Metal")

plt.savefig("ae.png")