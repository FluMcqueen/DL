'''
Audio par
- num canali (1 o 2)
- sample width (num bytes)
- framerate (num samp al secondo)
- tot frames
- values of a frame (binary)
'''

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

plt.savefig("tutwav.png")