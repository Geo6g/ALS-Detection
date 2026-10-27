"""
Ultrasound Model Module
Implements Convolutional Neural Network (CNN) architecture for muscle B-mode ultrasound classification:
- Explicit named convolutional stages for Grad-CAM visual interpretability ('conv_final')
- Latent feature extraction layer ('deep_features') for Multimodal Early Fusion
- Softmax probability estimation for ALS vs Healthy Control
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_ultrasound_cnn(input_shape: tuple = (224, 224, 3), num_classes: int = 2) -> tf.keras.Model:
    """
    Constructs a dedicated CNN architecture for muscle ultrasound classification.
    Layer naming is strictly preserved to ensure Grad-CAM and Early Fusion compatibility.
    """
    inputs = layers.Input(shape=input_shape, name='ultrasound_input')
    
    # Stage 1: Initial feature extraction
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_1')(x)
    
    # Stage 2: Mid-level texture & speckle patterns
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_2')(x)
    
    # Stage 3: High-level fascia & echogenicity morphology (Target for Grad-CAM)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_final')(x)
    x = layers.BatchNormalization(name='bn_final')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_final')(x)
    
    # Global spatial pooling & dense latent representation
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(64, activation='relu', name='deep_features')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    
    # Output class probabilities
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Ultrasound_ALS_CNN')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class UltrasoundClassifier:
    def __init__(self, input_shape: tuple = (224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.is_fitted = False

    def initialize_model(self):
        self.model = build_ultrasound_cnn(input_shape=self.input_shape)
        return self.model

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, batch_size: int = 16, validation_split: float = 0.2):
        """Trains the CNN on preprocessed ultrasound image tensors."""
        if self.model is None:
            self.initialize_model()
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        self.is_fitted = True
        return history

    def predict_proba(self, tensor: np.ndarray) -> np.ndarray:
        """Returns probability distribution [P(Control), P(ALS)]."""
        if self.model is None:
            raise ValueError("Ultrasound CNN model is not loaded or trained.")
        if len(tensor.shape) == 3:
            tensor = np.expand_dims(tensor, axis=0)
        return self.model.predict(tensor, verbose=0)

    def predict_single(self, tensor: np.ndarray) -> dict:
        """Predicts single image tensor and returns structured prediction."""
        probs = self.predict_proba(tensor)[0]
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

    def extract_deep_features(self, tensor: np.ndarray) -> np.ndarray:
        """Extracts the 64-dimensional latent embedding from 'deep_features' layer."""
        if self.model is None:
            raise ValueError("Ultrasound CNN model is not loaded.")
        if len(tensor.shape) == 3:
            tensor = np.expand_dims(tensor, axis=0)
        extractor = tf.keras.Model(
            inputs=self.model.input, 
            outputs=self.model.get_layer('deep_features').output
        )
        features = extractor.predict(tensor, verbose=0)
        return features.flatten()

    def save(self, filepath: str):
        if self.model is not None:
            self.model.save(filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True
        return self

def load_or_init_ultrasound_model(model_dir: str = 'models/saved') -> UltrasoundClassifier:
    """Loads saved model or returns initialized UltrasoundClassifier."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ultrasound_model.keras')
    clf = UltrasoundClassifier()
    if os.path.exists(model_path):
        try:
            clf.load(model_path)
            return clf
        except Exception:
            pass
    return clf
