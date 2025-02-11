"""
Takes in digital samples of a song/recording and produces a spectrogram of
log-scaled amplitudes and extract local peaks from it and takes the peaks from
the spectrogram and forms fingerprints via “fanout” patterns among the peaks.

Note: Only produce_spectrogram, extract_local_peak_idxs, and form_fingerprints need to be imported
"""
from audio_processing_config import configs as cfg
import numpy as np
from numba import njit
import matplotlib.mlab as mlab
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import iterate_structure
from typing import Tuple, List


def produce_spectrogram(samples: np.ndarray, sampling_rate: float = 44100):
    """
    Computes a spectrogram from audio samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Digital audio data

    sampling_rate : float
        fs - Number of samples taken per second

    Returns
    -------
    Tuple[spectrogram, freqs, times]
        spectrogram : numpy.ndarray, shape-(F,T)
            Array of amplitudes, in decibels

        freqs : numpy.ndarray, shape-(F,)
            Array of frequency values, in hertz (corresponds to y-axis)

        times : numpy.ndarray, shape-(T,)
            Array of time values, in seconds (corresponds to x-axis)
    """
    spectrogram, freqs, times = mlab.specgram(
        samples,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096/2)
    )
    spectrogram = np.log(np.clip(spectrogram, a_min=1e-20, a_max=None))
    return spectrogram, freqs, times


@njit
def _peaks(data_2d: np.ndarray, nbrhd_row_offsets: np.ndarray, nbrhd_col_offsets: np.ndarray, amp_min: float) \
        -> List[Tuple[int, int]]:
    """
    A Numba-optimized 2-D peak-finding algorithm.

    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    nbrhd_row_offsets : numpy.ndarray, shape-(N,)
        The row-index offsets used to traverse the local neighborhood.

        E.g., given the row/col-offsets (dr, dc), the element at
        index (r+dr, c+dc) will reside in the neighborhood centered at (r, c).

    nbrhd_col_offsets : numpy.ndarray, shape-(N,)
        The col-index offsets used to traverse the local neighborhood. See
        `nbrhd_row_offsets` for more details.

    amp_min : float
        All amplitudes equal to or below this value are excluded from being
        local peaks.

    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location, returned in
        column-major order
    """
    peaks = []

    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            continue
        for dr, dc in zip(nbrhd_row_offsets, nbrhd_col_offsets):
            if dr == 0 and dc == 0:
                continue
            if not (0 <= r + dr < data_2d.shape[0]):
                continue
            if not (0 <= c + dc < data_2d.shape[1]):
                continue
            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                break
        else:
            peaks.append((r, c))
    return peaks


def _local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.

    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected

    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers

    amp_min : float
        All amplitudes at and below this value are excluded from being local
        peaks.

    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location, returned
        in column-major ordering.

    Notes
    -----
    The local peaks are returned in column-major order, meaning that we
    iterate over all nbrhd_row_offsets in a given column of `data_2d` in search for
    local peaks, and then move to the next column.
    """
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    nbrhd_row_indices, nbrhd_col_indices = np.where(neighborhood)

    nbrhd_row_offsets = nbrhd_row_indices - neighborhood.shape[0] // 2
    nbrhd_col_offsets = nbrhd_col_indices - neighborhood.shape[1] // 2

    return _peaks(data_2d, nbrhd_row_offsets, nbrhd_col_offsets, amp_min=amp_min)


def _find_cutoff(spectrogram: np.ndarray) -> float:
    """
    Identifies minimum amplitude value to include when finding local peaks.

    Parameters
    ----------
    spectrogram : numpy.ndarray, shape-(F,T)
        Array of amplitudes, in decibels

    Returns
    -------
    float
        The amplitude at a given percentile.
    """
    data = spectrogram.ravel()  # flattens spectrogram
    idx = round(len(data) * cfg["amp_threshold_pct"])  # finds index of amp threshold percentile
    return np.partition(data, idx)[idx]


def extract_local_peak_idxs(spectrogram: np.ndarray):
    """
    Extracts index pairs of local peaks from a spectrogram.

    Parameters
    ----------
    spectrogram : numpy.ndarray, shape-(F,T)
        Array of amplitudes, in decibels

    Returns
    -------
    numpy.ndarray, shape-(N,2)
        (f,t) index pairs corresponding to each local peak
        location, N (# of local peaks) varies.

    Notes
    -----
    Index pairs map to (y,x) coordinate-wise.
    """
    # create neighborhood used to find peaks
    base_structure = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(base_structure, cfg["neighborhood_iterations"])
    # find value of minimum amp threshold (cutoff)
    cutoff = _find_cutoff(spectrogram)
    return np.array(_local_peak_locations(spectrogram, neighborhood, cutoff))


def _form_pair_encoding(peak_m: np.ndarray, peak_n: np.ndarray):
    """
    Encodes the relationship between two peaks.

    Parameters
    ----------
    peak_m : numpy.ndarray, shape-(2,)
        First peak, (f, t) value pair

    peak_n : numpy.ndarray, shape-(2,)
        Second peak, (f, t) value pair

    Returns
    -------
    Tuple[fm, fn, dt]
        fm : float
            Frequency value of first peak
        fn : float
            Frequency value of second peak
        dt : float
            Delta time value between two peaks (tn - tm)
    """
    return peak_m[0], peak_n[0], peak_n[1] - peak_m[1]


def form_fingerprints(local_peaks_idx: np.ndarray):
    """
    Forms the fingerprint for an audio recording.

    Parameters
    ----------
    local_peaks_idx : numpy.ndarray, shape-(N,2)
        (f,t) index pairs corresponding to each local peak location,
        N (# of local peaks) varies.

    Returns
    -------
    List[List[Tuple[int, int, int], int]]
        Aggregate of fanout patterns for all peaks in the audio
        recording which form the fingerprint for that recording.
    """
    fanout_size = cfg["fanout_size"]
    # uses local peak indexes to zip freqs and times into (t, f) peak point coordinate pairs
    fingerprints = []
    for i, idx in enumerate(local_peaks_idx):
        fanout = []
        fanout_local_peaks = local_peaks_idx[i + 1: i + fanout_size + 1]
        for close_peak in fanout_local_peaks:
            fanout.append([_form_pair_encoding(idx, close_peak), idx[1]])
        fingerprints.append(fanout)
    return fingerprints
