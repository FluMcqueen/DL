'''
Audio par
- num canali (1 o 2)
- sample width (num bytes)
- framerate (num samp al secondo)
- tot frames
- values of a frame (binary)

'''
import wave
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



wav_dir = "Benjamin_Netanyau/189.wav"
obj = wave.open(wav_dir, 'rb')

print("Number of channels", obj.getnchannels())
print("Sample", obj.getsampwidth()) #2 bytes for each sample
print("Frame Rate", obj.getframerate())
print("Number of frames", obj.getnframes())
print("-"*42)
print("Parmeters", obj.getparams())

tempo = obj.getnframes() / obj.getframerate()
print(tempo)

print("-"*42)

freq = obj.getframerate()
nsample = obj.getnframes()
signal_wave = obj.readframes(-1)

obj.close()

signal_array = np.frombuffer(signal_wave, dtype=np.int16)
time = np.linspace(0, tempo, num=nsample)

plt.figure(figsize=(15,5))
plt.plot(time, signal_array)
plt.title("Audio")
plt.ylabel("Signal Wave")
plt.xlabel("Time")
plt.xlim(0, tempo)
plt.show()

def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, freq)
    return audio

a = path_to_audio(wav_dir)
print(a)