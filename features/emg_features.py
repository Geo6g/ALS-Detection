"""
EMG Feature Extraction Module
Extracts time-domain and frequency-domain neuromuscular features from EMG signals:
- Root Mean Square (RMS)
- Mean Absolute Value (MAV)
- Zero Crossing Rate (ZCR)
- Waveform Length (WL)
- Variance (VAR)
- Slope Sign Changes (SSC)
- Integrated EMG (IEMG)
- Mean Frequency (MNF)
- Median Frequency (MDF)
- Peak Frequency (PKF)
- Total Spectral Power (TTP)
- Spectral Entropy (SE)
"""

import numpy as np
from scipy.signal import welch

def calc_rms(signal: np.ndarray) -> float:
    """Root Mean Square (RMS) reflecting motor unit recruitment."""
    return float(np.sqrt(np.mean(signal ** 2)))

def calc_mav(signal: np.ndarray) -> float:
    """Mean Absolute Value (MAV) representing average signal amplitude."""
    return float(np.mean(np.abs(signal)))

def calc_zcr(signal: np.ndarray, threshold: float = 0.01) -> float:
    """Zero Crossing Rate (ZCR) indicating signal frequency content."""
    diff = np.diff(np.sign(signal))
    # Count sign changes with magnitude above noise threshold
    valid = np.abs(np.diff(signal)) >= threshold
    crossings = np.sum((diff != 0) & valid)
    return float(crossings / len(signal))

def calc_waveform_length(signal: np.ndarray) -> float:
    """Cumulative waveform length (WL) measuring signal complexity."""
    return float(np.sum(np.abs(np.diff(signal))))

def calc_variance(signal: np.ndarray) -> float:
    """Variance (VAR) of the signal."""
    return float(np.var(signal))

def calc_ssc(signal: np.ndarray, threshold: float = 0.01) -> float:
    """Slope Sign Changes (SSC) measuring frequency of direction turns."""
    n = len(signal)
    if n < 3:
        return 0.0
    diff1 = signal[1:-1] - signal[:-2]
    diff2 = signal[1:-1] - signal[2:]
    changes = (diff1 * diff2) >= threshold
    return float(np.sum(changes) / (n - 2))

def calc_iemg(signal: np.ndarray) -> float:
    """Integrated EMG (IEMG) - total area under the rectified curve."""
    return float(np.sum(np.abs(signal)))

def calc_spectral_features(signal: np.ndarray, fs: float = 1000.0) -> dict:
    """
    Computes frequency domain features via Welch's Power Spectral Density (PSD):
    - Mean Frequency (MNF)
    - Median Frequency (MDF)
    - Peak Frequency (PKF)
    - Total Spectral Power (TTP)
    - Spectral Entropy (SE)
    """
    nperseg = min(len(signal), int(fs * 0.5)) if len(signal) >= 128 else len(signal)
    if nperseg < 16:
        nperseg = len(signal)
    
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_sum = np.sum(psd)
    
    if psd_sum <= 0 or np.isnan(psd_sum):
        return {
            'mnf': 0.0,
            'mdf': 0.0,
            'pkf': 0.0,
            'ttp': 0.0,
            'spectral_entropy': 0.0
        }
    
    # Mean Frequency (MNF)
    mnf = float(np.sum(freqs * psd) / psd_sum)
    
    # Median Frequency (MDF)
    cum_psd = np.cumsum(psd)
    mdf_idx = np.where(cum_psd >= 0.5 * psd_sum)[0]
    mdf = float(freqs[mdf_idx[0]]) if len(mdf_idx) > 0 else 0.0
    
    # Peak Frequency (PKF)
    pkf = float(freqs[np.argmax(psd)])
    
    # Total Power (TTP)
    ttp = float(psd_sum)
    
    # Spectral Entropy (SE)
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0.0
    
    return {
        'mnf': mnf,
        'mdf': mdf,
        'pkf': pkf,
        'ttp': ttp,
        'spectral_entropy': spectral_entropy
    }

FEATURE_NAMES = [
    'RMS',
    'MAV',
    'ZCR',
    'Waveform_Length',
    'Variance',
    'SSC',
    'IEMG',
    'Mean_Frequency',
    'Median_Frequency',
    'Peak_Frequency',
    'Total_Power',
    'Spectral_Entropy'
]

def extract_emg_feature_dict(signal: np.ndarray, fs: float = 1000.0) -> dict:
    """
    Extracts all EMG features and returns a named dictionary.
    """
    spec = calc_spectral_features(signal, fs=fs)
    return {
        'RMS': calc_rms(signal),
        'MAV': calc_mav(signal),
        'ZCR': calc_zcr(signal),
        'Waveform_Length': calc_waveform_length(signal),
        'Variance': calc_variance(signal),
        'SSC': calc_ssc(signal),
        'IEMG': calc_iemg(signal),
        'Mean_Frequency': spec['mnf'],
        'Median_Frequency': spec['mdf'],
        'Peak_Frequency': spec['pkf'],
        'Total_Power': spec['ttp'],
        'Spectral_Entropy': spec['spectral_entropy']
    }

def extract_emg_features(signal: np.ndarray, fs: float = 1000.0) -> tuple:
    """
    Extracts all EMG features as a 1D numerical array along with feature names list.
    """
    f_dict = extract_emg_feature_dict(signal, fs=fs)
    vector = np.array([f_dict[name] for name in FEATURE_NAMES], dtype=np.float64)
    return vector, FEATURE_NAMES
