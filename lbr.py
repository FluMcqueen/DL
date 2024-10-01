import matplotlib.pyplot as plt
import numpy as np
import librosa as lbr
import soundfile as sf

wav_dir = "Benjamin_Netanyau/189.wav"

sig, sample_rate = sf.read(wav_dir)

print("Sample rate", sample_rate)

stf=lbr.stft(sig)
spec = np.abs(stf)
spec_db = lbr.amplitude_to_db(spec)

plt.figure(figsize=(15,7))
img = lbr.display.specshow(spec_db, y_axis="log", x_axis="time", sr=sample_rate, cmap="inferno")
plt.show()
