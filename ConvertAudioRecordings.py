# Creating functions for converting all variety of audio recordings, be them recorded from the microphone or digital audio files, into a NumPy-array of digital samples.

import numpy as np
from microphone import record_audio
import librosa

def samples_mic(listen_time) -> np.ndarray:

    frames, sample_rate = record_audio(int(listen_time))

    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])

    return samples, sample_rate


def samples_file(file_path: str):

    # `recorded_audio` is a numpy array of N audio samples
    recorded_audio, sampling_rate = librosa.load(file_path, sr=44100, mono=True)
    
    return recorded_audio, sampling_rate
