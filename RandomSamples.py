import numpy as np
import random

def random_samples(
    samples: np.ndarray,
    freq: int,
    n: int,
    length: float) -> np.ndarray:
    """
    Takes an array of audio samples from a long (e.g. one minute)
    recording and produce random clips of it at a desired, shorter length.
    This can help with experimentation/analysis. For example you can
    record a 1 minutes clip of a song, played from your phone and then
    create many random 10 second clips from it and see if they all
    successfully match against your database.

    Parameters
    ----------
    samples: numpy.ndarray, shape-(N,)
        The samples of the original recording
    freq: int
        Frequency of the recording
    n: int
        Number of random clips to generate
    length: float
        Length, in seconds, of each generated clip

    Returns
    ----------
    numpy.ndarray, shape(n, length * freq)
        An array of n arrays of samples, representing the n clips generated
    """
    original_number_of_samples = np.size(samples)
    samples_per_clip = int(length * freq)
    result = np.zeros((n, samples_per_clip))
    latest_start = original_number_of_samples - samples_per_clip

    for i in range(n):
        start = random.randint(0, latest_start)
        result[i] = samples[start:start + samples_per_clip]

    return result

if __name__ == "__main__":
    print(random_samples(np.arange(60000), 1000, 100, 10.0))
    print(np.shape(random_samples(np.arange(60000), 1000, 100, 10.0)))