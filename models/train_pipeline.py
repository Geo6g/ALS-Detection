"""
Training & Evaluation Pipeline
Orchestrates training across EMG, Ultrasound, Early Fusion, and Late Fusion models:
- Feature extraction & dataset splitting
- EMG Random Forest Classifier training
- Ultrasound CNN training
- Multimodal Early Fusion training
- Test set inference and benchmark metric computation
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing.emg_preprocessing import preprocess_emg_signal
from preprocessing.ultrasound_preprocessing import preprocess_ultrasound_image
from features.emg_features import extract_emg_features, FEATURE_NAMES
from features.ultrasound_features import extract_ultrasound_features
from models.emg_model import EMGClassifier
from models.ultrasound_model import UltrasoundClassifier
from models.fusion_model import EarlyFusionClassifier
from fusion.fusion_methods import early_fusion_combine, late_fusion_predict
from evaluation.metrics import calculate_evaluation_metrics
from data.dataset_loader import scan_emg_directory, scan_ultrasound_directory, load_emg_file, load_ultrasound_file

def run_multimodal_training_pipeline(data_dir: str = 'data/research_cohort', save_dir: str = 'models/saved', epochs_cnn: int = 12):
    """
    Executes end-to-end training and evaluation on available dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Scan datasets
    emg_scan = scan_emg_directory(os.path.join(data_dir, 'emg'))
    us_scan = scan_ultrasound_directory(os.path.join(data_dir, 'ultrasound'))

    if emg_scan['count'] == 0 or us_scan['count'] == 0:
        raise ValueError(f"Insufficient data in {data_dir}. EMG count: {emg_scan['count']}, US count: {us_scan['count']}")

    # Match paired samples by index or class
    min_count = min(emg_scan['count'], us_scan['count'])
    emg_samples = emg_scan['samples'][:min_count]
    us_samples = us_scan['images'][:min_count]

    # 2. Extract EMG Features
    X_emg = []
    y_labels = []
    for s in emg_samples:
        raw_sig = load_emg_file(s['path'])
        prep = preprocess_emg_signal(raw_sig)
        feats, _ = extract_emg_features(prep['processed'])
        X_emg.append(feats)
        y_labels.append(s['class_idx'])

    X_emg = np.array(X_emg)
    y_labels = np.array(y_labels)

    # 3. Load & Preprocess Ultrasound Images
    X_us_tensors = []
    for s in us_samples:
        img = load_ultrasound_file(s['path'])
        prep = preprocess_ultrasound_image(img, target_size=(224, 224))
        X_us_tensors.append(prep['tensor'][0])

    X_us_tensors = np.array(X_us_tensors)

    # Split into Train and Test sets (75% train, 25% test)
    indices = np.arange(len(y_labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=y_labels)

    X_emg_train, X_emg_test = X_emg[train_idx], X_emg[test_idx]
    X_us_train, X_us_test = X_us_tensors[train_idx], X_us_tensors[test_idx]
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]

    # --- Train EMG Model ---
    emg_clf = EMGClassifier()
    emg_clf.fit(X_emg_train, y_train, feature_names=FEATURE_NAMES)
    emg_clf.save(os.path.join(save_dir, 'emg_model.joblib'))

    # --- Train Ultrasound CNN ---
    us_clf = UltrasoundClassifier(input_shape=(224, 224, 3))
    us_clf.fit(X_us_train, y_train, epochs=epochs_cnn, batch_size=8, validation_split=0.15)
    us_clf.save(os.path.join(save_dir, 'ultrasound_model.keras'))

    # --- Extract Deep Latent Features for Early Fusion ---
    us_deep_train = []
    for tensor in X_us_train:
        feat = us_clf.extract_deep_features(tensor)
        us_deep_train.append(feat)
    us_deep_train = np.array(us_deep_train)

    us_deep_test = []
    for tensor in X_us_test:
        feat = us_clf.extract_deep_features(tensor)
        us_deep_test.append(feat)
    us_deep_test = np.array(us_deep_test)

    # Combine for Early Fusion
    X_early_train = np.hstack([X_emg_train, us_deep_train])
    X_early_test = np.hstack([X_emg_test, us_deep_test])

    early_clf = EarlyFusionClassifier()
    early_clf.fit(X_early_train, y_train)
    early_clf.save(os.path.join(save_dir, 'early_fusion_model.joblib'))

    # --- Evaluate All Modalities on Held-Out Test Set ---
    # 1. EMG Only
    emg_probs = emg_clf.predict_proba(X_emg_test)
    emg_preds = (emg_probs[:, 1] >= 0.5).astype(int)
    emg_metrics = calculate_evaluation_metrics(y_test, emg_preds, emg_probs)

    # 2. Ultrasound Only
    us_probs = us_clf.predict_proba(X_us_test)
    us_preds = (us_probs[:, 1] >= 0.5).astype(int)
    us_metrics = calculate_evaluation_metrics(y_test, us_preds, us_probs)

    # 3. Early Fusion
    early_probs = early_clf.predict_proba(X_early_test)
    early_preds = (early_probs[:, 1] >= 0.5).astype(int)
    early_metrics = calculate_evaluation_metrics(y_test, early_preds, early_probs)

    # 4. Late Fusion (alpha = 0.5 default)
    late_probs_list = []
    late_preds_list = []
    for ep, up in zip(emg_probs[:, 1], us_probs[:, 1]):
        res = late_fusion_predict(ep, up, alpha=0.5)
        late_probs_list.append([1.0 - res['fused_score'], res['fused_score']])
        late_preds_list.append(res['class_index'])
    
    late_probs = np.array(late_probs_list)
    late_preds = np.array(late_preds_list)
    late_metrics = calculate_evaluation_metrics(y_test, late_preds, late_probs)

    # Save test data for runtime evaluation
    np.savez(
        os.path.join(save_dir, 'test_evaluation_cache.npz'),
        y_test=y_test,
        emg_probs=emg_probs,
        emg_preds=emg_preds,
        us_probs=us_probs,
        us_preds=us_preds,
        early_probs=early_probs,
        early_preds=early_preds,
        late_probs=late_probs,
        late_preds=late_preds,
        X_emg_test=X_emg_test,
        X_emg_train=X_emg_train
    )

    return {
        'emg_clf': emg_clf,
        'us_clf': us_clf,
        'early_clf': early_clf,
        'metrics': {
            'EMG Model': emg_metrics,
            'Ultrasound Model': us_metrics,
            'Early Fusion': early_metrics,
            'Late Fusion (alpha=0.5)': late_metrics
        },
        'test_count': len(y_test)
    }
