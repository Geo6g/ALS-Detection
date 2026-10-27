"""
Early Fusion Model Module
Trains and performs inference on joint multimodal feature representations:
z = [ f_EMG || f_Ultrasound ]
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

class EarlyFusionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        base_clf = GradientBoostingClassifier(
            n_estimators=120, 
            learning_rate=0.05, 
            max_depth=4, 
            random_state=42
        )
        self.model = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
        self.is_fitted = False

    def fit(self, X_joint: np.ndarray, y: np.ndarray):
        """Fits joint multimodal classifier on concatenated feature vectors."""
        X_scaled = self.scaler.fit_transform(X_joint)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X_joint: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Early Fusion model is not trained yet.")
        if len(X_joint.shape) == 1:
            X_joint = X_joint.reshape(1, -1)
        X_scaled = self.scaler.transform(X_joint)
        return self.model.predict_proba(X_scaled)

    def predict(self, X_joint: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_joint)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_single(self, joint_vector: np.ndarray) -> dict:
        probs = self.predict_proba(joint_vector)[0]
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
            'is_fitted': self.is_fitted
        }, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.model = data['model']
        self.is_fitted = data['is_fitted']
        return self

def load_or_init_fusion_model(model_dir: str = 'models/saved') -> EarlyFusionClassifier:
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'early_fusion_model.joblib')
    clf = EarlyFusionClassifier()
    if os.path.exists(model_path):
        try:
            clf.load(model_path)
            return clf
        except Exception:
            pass
    return clf
