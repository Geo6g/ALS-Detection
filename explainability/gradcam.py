"""
Grad-CAM Explainability Module for Ultrasound CNN
Implements Gradient-weighted Class Activation Mapping (Grad-CAM):
- Extracts spatial activation maps from the final convolutional layer ('conv_final')
- Calculates analytical gradients of the ALS class probability with respect to feature maps
- Generates high-resolution heatmaps localized on muscle fascial and echogenicity anomalies
- Superimposes heatmaps over original B-mode ultrasound images
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def compute_gradcam_heatmap(
    model: tf.keras.Model, 
    image_tensor: np.ndarray, 
    target_layer_name: str = 'conv_final', 
    pred_index: int = None
) -> np.ndarray:
    """
    Computes 2D Grad-CAM activation heatmap normalized in [0, 1].
    """
    if model is None:
        raise ValueError("Grad-CAM explanation cannot be generated: Ultrasound CNN model is not loaded.")

    if len(image_tensor.shape) == 3:
        image_tensor = np.expand_dims(image_tensor, axis=0)

    # Find the target layer
    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError:
        # Fallback to the last 4D Conv2D layer in the model
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError(f"No convolutional layers found in the model for Grad-CAM.")
        target_layer_name = conv_layers[-1]
        target_layer = model.get_layer(target_layer_name)

    # Build gradient extraction sub-model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Calculate gradients of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight convolutional output maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only positive contributions toward target class
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_overlay(
    original_img: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.4, 
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Superimposes 2D heatmap over the original ultrasound image.
    """
    # Ensure original is uint8 RGB [0, 255]
    if original_img.dtype != np.uint8:
        if np.max(original_img) <= 1.0:
            orig = (original_img * 255).astype(np.uint8)
        else:
            orig = original_img.astype(np.uint8)
    else:
        orig = original_img.copy()

    if len(orig.shape) == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    h, w = orig.shape[:2]
    # Resize heatmap to match image dimensions
    resized_heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(resized_heatmap, 0.0, 1.0))

    # Apply color map
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Blend images
    superimposed = cv2.addWeighted(colored_heatmap, alpha, orig, 1.0 - alpha, 0)
    return superimposed, colored_heatmap

def explain_ultrasound_gradcam(
    model_obj, 
    original_img: np.ndarray, 
    tensor_input: np.ndarray, 
    target_layer_name: str = 'conv_final',
    class_idx: int = 1,
    alpha: float = 0.45
) -> dict:
    """
    Full pipeline to compute Grad-CAM heatmap and visualization overlay.
    """
    raw_model = getattr(model_obj, 'model', model_obj)
    if raw_model is None:
        raise ValueError("Grad-CAM explanation cannot be generated: Ultrasound model is not trained or connected.")

    heatmap = compute_gradcam_heatmap(
        raw_model, 
        tensor_input, 
        target_layer_name=target_layer_name, 
        pred_index=class_idx
    )
    
    superimposed, colored_heatmap = generate_gradcam_overlay(
        original_img, 
        heatmap, 
        alpha=alpha
    )

    # Matplotlib summary figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_img)
    axes[0].set_title('Original Ultrasound', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap (conv_final)', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(superimposed)
    axes[2].set_title('Superimposed Attributions', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()

    return {
        'heatmap': heatmap,
        'superimposed': superimposed,
        'colored_heatmap': colored_heatmap,
        'fig': fig
    }
