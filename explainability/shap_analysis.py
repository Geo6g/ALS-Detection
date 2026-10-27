"""
SHAP Explainability Module for EMG Features
Computes Shapley additive explanations (SHAP) for EMG neuromuscular feature contributions:
- TreeExplainer / LinearExplainer / KernelExplainer
- Summary attribution bar chart
- Sample-level Waterfall / Force attribution plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def explain_emg_sample_shap(
    model_obj, 
    sample_vector: np.ndarray, 
    background_data: np.ndarray = None, 
    feature_names: list = None
):
    """
    Computes exact or kernel SHAP values for a single EMG feature vector.
    Returns:
    - shap_values (1D array of importance for the sample)
    - base_value (expected value)
    - fig_bar (matplotlib figure of feature contributions)
    - fig_waterfall (waterfall plot figure)
    - df_contributions (pandas DataFrame sorted by absolute impact)
    """
    if model_obj is None or not getattr(model_obj, 'is_fitted', False):
        raise ValueError("SHAP explanation cannot be generated: EMG model is not trained or connected.")

    if feature_names is None:
        feature_names = model_obj.feature_names or [f"Feature_{i}" for i in range(len(sample_vector))]

    # Get underlying sklearn model and scaler
    scaler = model_obj.scaler
    underlying = model_obj.model
    
    # Extract estimator (handle CalibratedClassifierCV or raw classifier)
    if hasattr(underlying, 'calibrated_classifiers_') and len(underlying.calibrated_classifiers_) > 0:
        base_estimator = underlying.calibrated_classifiers_[0].estimator
    elif hasattr(underlying, 'estimator'):
        base_estimator = underlying.estimator
    else:
        base_estimator = underlying

    X_scaled_sample = scaler.transform(sample_vector.reshape(1, -1))
    
    # Build explainer
    try:
        explainer = shap.TreeExplainer(base_estimator)
        raw_shap = explainer.shap_values(X_scaled_sample)
        
        # In multi-class, select class 1 (ALS)
        if isinstance(raw_shap, list):
            sample_shap = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
            base_val = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        elif isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
            sample_shap = raw_shap[0, :, 1]
            base_val = float(explainer.expected_value[1])
        else:
            sample_shap = raw_shap[0] if len(raw_shap.shape) > 1 else raw_shap
            base_val = float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0])
    except Exception:
        # Fallback to KernelExplainer with background data
        if background_data is None:
            bg_scaled = np.zeros((10, len(sample_vector)))
        else:
            bg_scaled = scaler.transform(background_data[:20])
        explainer = shap.KernelExplainer(lambda x: model_obj.predict_proba(scaler.inverse_transform(x))[:, 1], bg_scaled)
        sample_shap = explainer.shap_values(X_scaled_sample)[0]
        base_val = float(explainer.expected_value)

    # 1. Feature contribution DataFrame
    df_contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': sample_vector,
        'SHAP_Value': sample_shap,
        'Absolute_Impact': np.abs(sample_shap),
        'Direction': ['Increases ALS Risk' if v > 0 else 'Decreases ALS Risk' for v in sample_shap]
    }).sort_values(by='Absolute_Impact', ascending=False).reset_index(drop=True)

    # 2. Matplotlib Bar Plot for local attribution
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_bar, ax = plt.subplots(figsize=(8, 5))
    top_df = df_contributions.head(10).iloc[::-1]
    colors = ['#c0392b' if val > 0 else '#2980b9' for val in top_df['SHAP_Value']]
    
    ax.barh(top_df['Feature'], top_df['SHAP_Value'], color=colors, height=0.6)
    ax.axvline(0, color='#333333', linewidth=0.8, linestyle='--')
    ax.set_xlabel('SHAP Value (Impact on Model Log-Odds for ALS)', fontsize=10)
    ax.set_title('EMG Feature Importance Attribution (SHAP)', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()

    # 3. Waterfall Plot
    fig_waterfall = None
    try:
        explanation_obj = shap.Explanation(
            values=sample_shap,
            base_values=base_val,
            data=sample_vector,
            feature_names=feature_names
        )
        fig_waterfall, ax_wf = plt.subplots(figsize=(8, 5.5))
        shap.plots.waterfall(explanation_obj, max_display=8, show=False)
        plt.tight_layout()
    except Exception:
        fig_waterfall = None

    return {
        'shap_values': sample_shap,
        'base_value': base_val,
        'fig_bar': fig_bar,
        'fig_waterfall': fig_waterfall,
        'table': df_contributions
    }
