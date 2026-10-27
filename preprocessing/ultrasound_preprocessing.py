"""
Ultrasound Preprocessing Module
Implements image processing pipelines for muscle B-mode ultrasound:
- Image loading & color space alignment
- Spatial resizing to CNN input dimensions
- CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- Intensity normalization
"""

import cv2
import numpy as np
from PIL import Image
import io

def load_ultrasound_image(input_data, target_channels: int = 3) -> np.ndarray:
    """
    Loads ultrasound image from filepath, byte buffer, PIL Image, or ndarray.
    """
    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise FileNotFoundError(f"Could not read image file at {input_data}")
        if target_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    elif isinstance(input_data, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(input_data))
        img = np.array(pil_img)
        if target_channels == 3 and (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif target_channels == 1 and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    elif isinstance(input_data, Image.Image):
        img = np.array(input_data)
        if target_channels == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif target_channels == 1 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    elif isinstance(input_data, np.ndarray):
        img = input_data.copy()
        if target_channels == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif target_channels == 1 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    else:
        raise TypeError(f"Unsupported image input type: {type(input_data)}")

def resize_image(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resizes image to target spatial dimensions (height, width).
    """
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE for local contrast enhancement of muscle fascial planes and speckle structure.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 2:
        return clahe.apply(img)
    elif len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return img

def normalize_image(img: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Scales pixel values to [0.0, 1.0] float32.
    """
    img_float = img.astype(np.float32) / 255.0
    if method == 'imagenet' and len(img.shape) == 3 and img.shape[2] == 3:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (img_float - mean) / std
    return img_float

def preprocess_ultrasound_image(
    input_data, 
    target_size: tuple = (224, 224),
    apply_enhancement: bool = True,
    norm_method: str = 'standard'
) -> dict:
    """
    Executes the complete ultrasound image preprocessing pipeline.
    """
    original = load_ultrasound_image(input_data, target_channels=3)
    resized = resize_image(original, target_size=target_size)
    enhanced = apply_clahe(resized) if apply_enhancement else resized
    normalized = normalize_image(enhanced, method=norm_method)
    tensor = np.expand_dims(normalized, axis=0)

    return {
        'original': original,
        'resized': resized,
        'enhanced': enhanced,
        'normalized': normalized,
        'tensor': tensor,
        'dimensions': original.shape
    }
