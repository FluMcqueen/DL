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

'''
plt.figure(figsize=(15,17))

plt.subplot(2,1,1)
librosa.display.waveshow(classical, alpha=0.5)
plt.ylim((-1, 1))
plt.title("Class")

plt.subplot(2,1,2)
librosa.display.waveshow(metal, alpha=0.5)
plt.ylim((-1, 1))
plt.title("Metal")

plt.show()

plt.savefig("tutwav.png")
'''

FRAME_SIZE = 2048
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

'''
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
'''

rms_clas = lbr.feature.rms(y = classical, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
rms_metal = lbr.feature.rms(y = metal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

framesmr = range(len(rms_metal))
tmr = librosa.frames_to_time(framesmr, hop_length=HOP_LENGTH)

'''
plt.figure(figsize=(15, 17))

ax = plt.subplot(2, 1, 1)
librosa.display.waveshow(classical, alpha=0.5)
plt.plot(tcl, rms_clas, color="r")
plt.ylim((-1, 1))
plt.title("Classical")

plt.subplot(2, 1, 2)
librosa.display.waveshow(metal, alpha=0.5)
plt.plot(tmr, rms_metal, color="r")
plt.ylim((-1, 1))
plt.title("Metal")

plt.savefig("rms.png")
'''

zcr_classical = librosa.feature.zero_crossing_rate(classical, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_metal = librosa.feature.zero_crossing_rate(metal, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

sp_classical = lbr.stft(classical, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
sp_metal = lbr.stft(metal, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)

print("-"*42)
print(sp_classical.shape) #nfreqency bins, nframes
print(sp_metal.shape)

Y_scale = np.abs(sp_classical) ** 2
Y_classical = lbr.power_to_db(Y_scale)

Y_scale = np.abs(sp_metal) ** 2
Y_metal = lbr.power_to_db(Y_scale)

def plot (Y, sr, hl, nome):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y, y_axis="log", x_axis="time", sr=sr, hop_length=hl)
    plt.colorbar(format="%+2.f")
    plt.savefig(f"sp_{nome}.png")

'''
plot(Y_classical, sr, HOP_LENGTH, "classical")
plot(Y_metal, sr, HOP_LENGTH, "metal")


filt = lbr.filters.mel(sr=sr, n_fft=FRAME_SIZE)

plt.figure(figsize=(25,10))
librosa.display.specshow(filt, sr=sr, x_axis="linear")
plt.colorbar(format="%+2.f")
plt.savefig(f"mspec.png")
'''

mspec_clas = lbr.feature.melspectrogram(y=classical, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)
mspec_metal = lbr.feature.melspectrogram(y=metal, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)

mspec_clas = lbr.power_to_db(mspec_clas)
mspec_metal = lbr.power_to_db(mspec_metal)

'''
def mplot (Y, sr, hl, nome):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y, y_axis="mel", x_axis="time", sr=sr, hop_length=hl, cmap="inferno")
    plt.colorbar(format="%+2.f")
    plt.savefig(f"msp_{nome}.png")


mplot(mspec_clas, sr=sr, hl=HOP_LENGTH, nome="classical")
mplot(mspec_metal, sr=sr, hl=HOP_LENGTH, nome="metal")
'''

mfc_clas = lbr.feature.mfcc(y=classical, sr=sr, n_mfcc=13)
mfc_metal = lbr.feature.mfcc(y=metal, sr=sr, n_mfcc=13)

def mfccplot (Y, sr, nome):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y, x_axis="time", sr=sr, cmap="inferno")
    plt.colorbar(format="%+2.f")
    plt.savefig(f"mfcc_{nome}.png")


mfccplot(mfc_clas, sr=sr, nome="classical")
mfccplot(mfc_metal, sr=sr, nome="metal")

delta_clas = lbr.feature.delta(mfc_clas)
delta2_clas = lbr.feature.delta(mfc_clas, order=2)
delta_met = lbr.feature.delta(mfc_metal)
delta2_met = lbr.feature.delta(mfc_metal, order=2)

print(delta_clas.shape)

comp_delta_clas = np.concatenate({mfc_clas, delta_clas, delta2_clas})
comp_delta_metal = np.concatenate({mfc_metal, delta_met, delta2_met})