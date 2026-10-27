"""
EMG Preprocessing Module
Implements biomedical signal processing pipelines for Electromyography signals:
- Butterworth Bandpass filtering (20 - 450 Hz)
- IIR Notch filtering (50 / 60 Hz powerline cancellation)
- Full-wave Rectification & Linear Envelope Extraction
- Amplitude Normalization (Z-score and Min-Max)
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def butter_bandpass_filter(data: np.ndarray, lowcut: float = 20.0, highcut: float = 450.0, fs: float = 1000.0, order: int = 4) -> np.ndarray:
    """
    Applies a zero-phase 4th-order Butterworth bandpass filter.
    Standard EMG physiological passband is between 20 Hz and 450 Hz.
    """
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if low >= high:
        raise ValueError(f"Invalid filter cutoff frequencies: low ({lowcut} Hz) >= high ({highcut} Hz)")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data: np.ndarray, notch_freq: float = 50.0, fs: float = 1000.0, quality_factor: float = 30.0) -> np.ndarray:
    """
    Applies an IIR notch filter to remove power-line interference (50 Hz or 60 Hz).
    """
    nyq = 0.5 * fs
    if notch_freq >= nyq or notch_freq <= 0:
        return data
    w0 = notch_freq / nyq
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, data)

def rectify_signal(data: np.ndarray) -> np.ndarray:
    """
    Full-wave rectification of the EMG signal.
    """
    return np.abs(data)

def linear_envelope(data: np.ndarray, cutoff: float = 10.0, fs: float = 1000.0, order: int = 4) -> np.ndarray:
    """
    Extracts the smooth linear envelope of an EMG signal using lowpass filtering of the rectified waveform.
    """
    nyq = 0.5 * fs
    norm_cutoff = min(0.999, cutoff / nyq)
    b, a = butter(order, norm_cutoff, btype='low')
    rectified = rectify_signal(data)
    return filtfilt(b, a, rectified)

def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalizes the signal amplitude:
    - 'zscore': zero-mean and unit-variance
    - 'minmax': scaled to [0, 1]
    """
    if method == 'zscore':
        std = np.std(data)
        if std == 0:
            return data - np.mean(data)
        return (data - np.mean(data)) / std
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    return data

def preprocess_emg_signal(
    signal: np.ndarray, 
    fs: float = 1000.0, 
    lowcut: float = 20.0, 
    highcut: float = 450.0, 
    notch_freq: float = 50.0,
    normalize: bool = True,
    norm_method: str = 'zscore'
) -> dict:
    """
    Executes the complete EMG preprocessing pipeline.
    """
    raw = np.asarray(signal, dtype=np.float64)
    bp = butter_bandpass_filter(raw, lowcut=lowcut, highcut=highcut, fs=fs)
    notched = notch_filter(bp, notch_freq=notch_freq, fs=fs) if notch_freq > 0 else bp
    env = linear_envelope(notched, cutoff=10.0, fs=fs)
    final_processed = normalize_signal(notched, method=norm_method) if normalize else notched

    return {
        'raw': raw,
        'bandpass': bp,
        'notched': notched,
        'envelope': env,
        'processed': final_processed,
        'sampling_rate': fs
    }
