"""
Multimodal Fusion Module
Implements Early (Feature-Level) and Late (Decision-Level) Multimodal Fusion:
1. Early Fusion: Feature concatenation [ f_EMG || f_Ultrasound ]
2. Late Fusion: Weighted Probability Blending:
   Final Score = alpha * P(ALS|EMG) + (1 - alpha) * P(ALS|US)
"""

import numpy as np
import pandas as pd

def early_fusion_combine(emg_features: np.ndarray, ultrasound_features: np.ndarray) -> np.ndarray:
    """
    Concatenates 1D EMG feature vector and 1D ultrasound feature vector into a unified joint vector.
    """
    emg_flat = np.asarray(emg_features, dtype=np.float64).flatten()
    us_flat = np.asarray(ultrasound_features, dtype=np.float64).flatten()
    return np.concatenate([emg_flat, us_flat])

def late_fusion_predict(emg_als_prob: float, us_als_prob: float, alpha: float = 0.5, threshold: float = 0.5) -> dict:
    """
    Computes Late Fusion decision score via linear probability interpolation:
    Score = alpha * P(ALS|EMG) + (1 - alpha) * P(ALS|Ultrasound)
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    emg_p = float(np.clip(emg_als_prob, 0.0, 1.0))
    us_p = float(np.clip(us_als_prob, 0.0, 1.0))
    
    fused_score = alpha * emg_p + (1.0 - alpha) * us_p
    prediction = "ALS" if fused_score >= threshold else "Healthy Control"
    
    return {
        'emg_probability': emg_p,
        'ultrasound_probability': us_p,
        'alpha': alpha,
        'fused_score': fused_score,
        'prediction': prediction,
        'class_index': int(fused_score >= threshold),
        'confidence': max(fused_score, 1.0 - fused_score),
        'formula_str': f"{alpha:.2f} × {emg_p:.3f} + {1.0 - alpha:.2f} × {us_p:.3f} = {fused_score:.3f}"
    }

def format_modal_comparison_table(
    emg_result: dict,
    us_result: dict,
    early_result: dict,
    late_result: dict
) -> pd.DataFrame:
    """
    Creates a clean tabular comparison across all four modal configurations.
    """
    rows = [
        {
            'Modality / Method': 'EMG Only (Functional)',
            'Prediction': emg_result.get('prediction', 'N/A'),
            'ALS Probability': f"{emg_result.get('als_probability', 0.0):.1%}",
            'Confidence': f"{emg_result.get('confidence', 0.0):.1%}",
            'Description': 'Time & frequency features from motor unit action potentials'
        },
        {
            'Modality / Method': 'Ultrasound Only (Structural)',
            'Prediction': us_result.get('prediction', 'N/A'),
            'ALS Probability': f"{us_result.get('als_probability', 0.0):.1%}",
            'Confidence': f"{us_result.get('confidence', 0.0):.1%}",
            'Description': 'CNN deep features & muscle echogenicity patterns'
        },
        {
            'Modality / Method': 'Early Fusion (Feature-Level)',
            'Prediction': early_result.get('prediction', 'N/A'),
            'ALS Probability': f"{early_result.get('als_probability', 0.0):.1%}",
            'Confidence': f"{early_result.get('confidence', 0.0):.1%}",
            'Description': 'Joint representation [f_EMG || f_US] with multimodal classifier'
        },
        {
            'Modality / Method': 'Late Fusion (Decision-Level)',
            'Prediction': late_result.get('prediction', 'N/A'),
            'ALS Probability': f"{late_result.get('fused_score', 0.0):.1%}",
            'Confidence': f"{late_result.get('confidence', 0.0):.1%}",
            'Description': f"Weighted blend with alpha = {late_result.get('alpha', 0.5):.2f}"
        }
    ]
    return pd.DataFrame(rows)
