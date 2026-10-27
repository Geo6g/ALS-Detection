"""
Ultrasound Feature Extraction Module
Extracts quantitative echogenicity, texture, and deep latent representations from muscle B-mode ultrasound:
- Mean Gray Level (MGL / Echogenicity)
- Echogenicity Standard Deviation (Speckle Heterogeneity)
- Pixel Distribution Skewness & Kurtosis
- High-Echogenicity Pixel Fraction (Fibrosis / Fatty infiltration biomarker)
- Spatial Texture Metrics (Contrast, Homogeneity, Energy, Entropy)
- Deep CNN Latent Embedding Extractor
"""

import cv2
import numpy as np
from scipy.stats import skew, kurtosis

ULTRASOUND_FEATURE_NAMES = [
    'Mean_Echogenicity',
    'Echogenicity_Std',
    'Skewness',
    'Kurtosis',
    'High_Echo_Ratio',
    'Texture_Contrast',
    'Texture_Homogeneity',
    'Texture_Energy',
    'Texture_Entropy'
]

def extract_echogenicity_features(img: np.ndarray) -> dict:
    """
    Extracts statistical and texture echogenicity parameters from ultrasound image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    flat = gray.astype(np.float64).flatten()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    skew_val = float(skew(flat)) if std_val > 0 else 0.0
    kurt_val = float(kurtosis(flat)) if std_val > 0 else 0.0
    
    # High echogenicity ratio (pixels above 60% intensity threshold ~ 153/255)
    high_echo = float(np.mean(flat >= 153.0))

    # Fast spatial co-occurrence texture approximations (dx=1, dy=0)
    diff_h = np.abs(gray[:, 1:].astype(np.float64) - gray[:, :-1].astype(np.float64))
    texture_contrast = float(np.mean(diff_h ** 2))
    texture_homogeneity = float(np.mean(1.0 / (1.0 + diff_h)))
    
    # Normalized histogram energy and entropy
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    texture_energy = float(np.sum(hist ** 2))
    texture_entropy = float(-np.sum(hist * np.log2(hist)))

    return {
        'Mean_Echogenicity': mean_val,
        'Echogenicity_Std': std_val,
        'Skewness': skew_val,
        'Kurtosis': kurt_val,
        'High_Echo_Ratio': high_echo,
        'Texture_Contrast': texture_contrast,
        'Texture_Homogeneity': texture_homogeneity,
        'Texture_Energy': texture_energy,
        'Texture_Entropy': texture_entropy
    }

def extract_ultrasound_features(img: np.ndarray) -> tuple:
    """
    Extracts ultrasound features as a 1D numerical array and feature names list.
    """
    f_dict = extract_echogenicity_features(img)
    vector = np.array([f_dict[k] for k in ULTRASOUND_FEATURE_NAMES], dtype=np.float64)
    return vector, ULTRASOUND_FEATURE_NAMES

def extract_deep_features(model, tensor: np.ndarray) -> np.ndarray:
    """
    Extracts deep latent feature vector from CNN intermediate representation layer.
    """
    if model is None:
        raise ValueError("CNN model is not loaded or trained.")
    
    import tensorflow as tf
    try:
        # Attempt to get 'deep_features' layer or global average pool output
        feature_layer = model.get_layer('deep_features')
        feature_extractor = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
        latent_features = feature_extractor.predict(tensor, verbose=0)
        return latent_features.flatten()
    except Exception:
        # Fallback to model output or penultimate layer
        penultimate = model.layers[-2]
        feature_extractor = tf.keras.Model(inputs=model.input, outputs=penultimate.output)
        latent_features = feature_extractor.predict(tensor, verbose=0)
        return latent_features.flatten()
