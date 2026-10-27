"""
EMG Model Module
Provides classification model and inference pipeline for EMG feature vectors.
Implements calibrated Random Forest / Gradient Boosting for ALS vs Healthy Control prediction.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

class EMGClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        base_rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=6, 
            random_state=42, 
            class_weight='balanced'
        )
        self.model = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """Fits the scaler and calibrated classifier."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.feature_names = feature_names
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability distribution [P(Control), P(ALS)]."""
        if not self.is_fitted:
            raise ValueError("EMG model is not trained yet.")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions (0 = Healthy Control, 1 = ALS)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_single(self, feature_vector: np.ndarray) -> dict:
        """Predicts a single feature vector and returns structured result."""
        probs = self.predict_proba(feature_vector)[0]
        als_prob = float(probs[1])
        control_prob = float(probs[0])
        pred_label = "ALS" if als_prob >= 0.5 else "Healthy Control"
        return {
            'prediction': pred_label,
            'class_index': int(als_prob >= 0.5),
            'als_probability': als_prob,
            'control_probability': control_prob,
            'confidence': max(als_prob, control_prob)
        }

    def save(self, filepath: str):
        joblib.dump({
            'scaler': self.scaler,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names
        }, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.model = data['model']
        self.is_fitted = data['is_fitted']
        self.feature_names = data.get('feature_names')
        return self

def load_or_train_default_emg_model(model_dir: str = 'models/saved') -> EMGClassifier:
    """Loads saved model or returns initialized EMGClassifier."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'emg_model.joblib')
    emg_clf = EMGClassifier()
    if os.path.exists(model_path):
        try:
            emg_clf.load(model_path)
            return emg_clf
        except Exception:
            pass
    return emg_clf
