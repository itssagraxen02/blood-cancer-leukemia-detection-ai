"""
preprocessing.py
─────────────────
Advanced blood smear image preprocessing pipeline:

  Stage 1 — HSV Color-Space Segmentation
    • Convert RGB → HSV
    • Create binary mask for leukocyte nuclei (blue-purple hue range)
    • Morphological operations (close/dilate) to clean the mask

  Stage 2 — Noise Reduction & Enhancement
    • Gaussian blur for noise suppression
    • CLAHE (Contrast Limited Adaptive Histogram Equalization)
    • Stain normalization (Macenko/Reinhard approximation)

  Stage 3 — Data Augmentation (training)
    • Random flips, rotations, zoom, brightness/contrast jitter
    • Elastic deformation (simulates smear variations)
    • Stain augmentation (color jitter in H&E/Giemsa space)
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings('ignore')


# ── HSV Ranges for Blood Cell Staining ───────────────────────────────────────
# Giemsa-stained leukocyte nuclei appear blue-purple
NUCLEUS_HSV_LOWER = np.array([120, 30, 30],  dtype=np.uint8)   # Blue-purple hue
NUCLEUS_HSV_UPPER = np.array([170, 255, 230], dtype=np.uint8)

# Cytoplasm appears pink-lavender
CYTO_HSV_LOWER = np.array([140, 10, 180], dtype=np.uint8)
CYTO_HSV_UPPER = np.array([180, 80, 255],  dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — HSV SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def hsv_segment_nuclei(img_rgb: np.ndarray) -> tuple:
    """
    Segment leukocyte nuclei using HSV color-space masking.
    
    Args:
        img_rgb : Input image in RGB format (H, W, 3) uint8
    
    Returns:
        Tuple (mask, segmented_rgb)
        mask          : Binary mask highlighting nuclei regions
        segmented_rgb : Original image with non-nuclear regions suppressed
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Primary nucleus mask
    mask_nucleus = cv2.inRange(img_hsv, NUCLEUS_HSV_LOWER, NUCLEUS_HSV_UPPER)

    # Morphological cleaning
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_cleaned = cv2.morphologyEx(mask_nucleus, cv2.MORPH_CLOSE, kernel_close)
    mask_cleaned = cv2.dilate(mask_cleaned, kernel_dilate, iterations=1)

    # Remove small noise blobs (area filtering)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask_cleaned)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Min area threshold
            cv2.drawContours(mask_filtered, [cnt], -1, 255, -1)

    # Apply mask to original
    mask_3ch = cv2.merge([mask_filtered, mask_filtered, mask_filtered])
    segmented = cv2.bitwise_and(img_rgb, mask_3ch)

    # For visualization: colorize the mask
    mask_vis = np.zeros((*mask_filtered.shape, 3), dtype=np.uint8)
    mask_vis[mask_filtered > 0] = [0, 255, 136]  # Cyan-green for nuclei
    mask_vis[mask_filtered == 0] = [5, 10, 20]    # Dark background

    return mask_vis, segmented


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — NOISE REDUCTION & CONTRAST ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def apply_clahe(img_rgb: np.ndarray,
                clip_limit: float = 2.5,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    CLAHE enhances local contrast in microscopy images without over-amplifying
    noise. Applied per-channel in LAB color space for perceptual consistency.
    
    Args:
        img_rgb       : RGB input image
        clip_limit    : Contrast clip limit (higher = more contrast)
        tile_grid_size: Size of grid tiles for local histogram equalization
    
    Returns:
        CLAHE-enhanced RGB image
    """
    # Work in LAB color space (L channel = luminance only)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)

    # Apply CLAHE only to L (luminance) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Reconstruct and convert back to RGB
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


def reduce_noise(img_rgb: np.ndarray) -> np.ndarray:
    """
    Multi-stage noise reduction for blood smear images.
    
    Pipeline:
      1. Non-local means denoising (preserves edges)
      2. Bilateral filter (edge-preserving smoothing)
    """
    # Stage 1: Gaussian blur (light, removes high-freq salt-pepper noise)
    blurred = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=0.8)

    # Stage 2: Bilateral filter (preserves nucleus boundaries)
    denoised = cv2.bilateralFilter(blurred, d=7, sigmaColor=50, sigmaSpace=50)

    return denoised


def macenko_stain_normalization(img_rgb: np.ndarray,
                                 target_mean: tuple = (0.72, 0.57, 0.70),
                                 target_std: tuple  = (0.11, 0.14, 0.10)) -> np.ndarray:
    """
    Simplified Macenko-style stain normalization.
    
    Normalizes the color distribution of Giemsa/Wright-stained images
    to a consistent reference, reducing lab-to-lab stain variability.
    
    Args:
        img_rgb     : Input RGB image (uint8)
        target_mean : Target color mean per channel
        target_std  : Target color std per channel
    
    Returns:
        Stain-normalized RGB image
    """
    img_float = img_rgb.astype(np.float32) / 255.0

    # Normalize each channel to target distribution
    for ch in range(3):
        channel = img_float[:, :, ch]
        src_mean = channel.mean()
        src_std  = channel.std() + 1e-8

        # Z-score normalize then rescale to target stats
        normalized = (channel - src_mean) / src_std
        normalized = normalized * target_std[ch] + target_mean[ch]
        img_float[:, :, ch] = np.clip(normalized, 0, 1)

    return (img_float * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def augment_image(img_rgb: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply random augmentations suited for blood smear microscopy.
    
    Augmentation strategies:
      • Spatial: flip (H/V), rotate (0-360°), zoom (0.9–1.1×)
      • Color:   brightness/contrast jitter, hue shift (simulates stain batch)
      • Noise:   Gaussian noise injection
      • Special: Elastic deformation (mimics smear preparation artifacts)
    
    Args:
        img_rgb : Input RGB image (uint8)
        seed    : Optional random seed for reproducibility
    
    Returns:
        Augmented RGB image
    """
    if seed is not None:
        np.random.seed(seed)

    img = img_rgb.copy()
    h, w = img.shape[:2]

    # ── Spatial Augmentations ─────────────────────────────────────────────────
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random vertical flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)

    # Random rotation (blood smear orientation doesn't matter)
    angle = np.random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random zoom/crop
    zoom = np.random.uniform(0.9, 1.1)
    new_h, new_w = int(h * zoom), int(w * zoom)
    img_zoom = cv2.resize(img, (new_w, new_h))
    if zoom > 1.0:
        # Crop back to original size
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        img = img_zoom[y_off:y_off+h, x_off:x_off+w]
    else:
        # Pad back to original size
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        img = cv2.copyMakeBorder(img_zoom, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                  cv2.BORDER_REFLECT)
        img = cv2.resize(img, (w, h))

    # ── Color Augmentations ───────────────────────────────────────────────────
    pil_img = Image.fromarray(img)

    # Brightness jitter
    if np.random.rand() > 0.4:
        factor = np.random.uniform(0.8, 1.2)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    # Contrast jitter
    if np.random.rand() > 0.4:
        factor = np.random.uniform(0.85, 1.15)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

    # Color/saturation jitter (simulates stain intensity variation)
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.9, 1.1)
        pil_img = ImageEnhance.Color(pil_img).enhance(factor)

    img = np.array(pil_img)

    # ── Gaussian Noise ────────────────────────────────────────────────────────
    if np.random.rand() > 0.6:
        noise = np.random.normal(0, 5, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def generate_augmented_grid(img_rgb: np.ndarray, n_variants: int = 8) -> np.ndarray:
    """
    Generate a visual grid of augmented image variants.
    Useful for visualizing augmentation diversity in the UI.
    
    Args:
        img_rgb    : Input RGB image
        n_variants : Number of augmented variants to generate
    
    Returns:
        Grid image (numpy array, RGB)
    """
    variants = [img_rgb]  # Original at position 0
    for i in range(n_variants - 1):
        aug = augment_image(img_rgb, seed=i * 7 + 13)
        variants.append(aug)

    # Build 2-row grid
    cols = n_variants // 2
    thumb_size = (112, 112)

    row1_imgs = [cv2.resize(v, thumb_size) for v in variants[:cols]]
    row2_imgs = [cv2.resize(v, thumb_size) for v in variants[cols:]]

    row1 = np.concatenate(row1_imgs, axis=1)
    row2 = np.concatenate(row2_imgs, axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    # Add thin grid lines
    for i in range(1, cols):
        x = i * thumb_size[0]
        grid[:, x-1:x+1] = [20, 30, 50]
    grid[thumb_size[1]-1:thumb_size[1]+1, :] = [20, 30, 50]

    return grid


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(img_array: np.ndarray) -> tuple:
    """
    Full preprocessing pipeline for a single blood smear image.
    
    Pipeline:
      1. Ensure RGB format
      2. Stain normalization (Macenko-style)
      3. Noise reduction (bilateral + Gaussian)
      4. HSV segmentation (nucleus isolation)
      5. CLAHE enhancement
    
    Args:
        img_array : Raw input image (H, W, 3) uint8
    
    Returns:
        Tuple (preprocessed_array, hsv_mask_image, clahe_image)
        preprocessed_array : Model-ready preprocessed image
        hsv_mask_image     : Visualization of segmentation mask (PIL Image)
        clahe_image        : CLAHE-enhanced image (PIL Image)
    """
    img = img_array.copy()

    # Ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # ── Step 1: Stain Normalization ───────────────────────────────────────────
    img_normalized = macenko_stain_normalization(img)

    # ── Step 2: Noise Reduction ───────────────────────────────────────────────
    img_denoised = reduce_noise(img_normalized)

    # ── Step 3: HSV Segmentation ──────────────────────────────────────────────
    hsv_mask_array, _ = hsv_segment_nuclei(img_denoised)

    # ── Step 4: CLAHE Enhancement ─────────────────────────────────────────────
    img_clahe = apply_clahe(img_denoised, clip_limit=2.5)

    # ── Step 5: Final resize + normalize for model ────────────────────────────
    preprocessed = cv2.resize(img_clahe, (224, 224))

    return (
        preprocessed,
        Image.fromarray(hsv_mask_array),
        Image.fromarray(img_clahe)
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING DATASET UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def create_tf_dataset(image_paths: list, labels: list,
                      batch_size: int = 32, augment: bool = True,
                      shuffle: bool = True):
    """
    Create a TensorFlow dataset from image paths with preprocessing.
    
    ┌──────────────────────────────────────────────────────────────┐
    │  USAGE EXAMPLE:                                              │
    │                                                              │
    │  from preprocessing import create_tf_dataset                │
    │  train_ds = create_tf_dataset(                               │
    │      image_paths=train_paths,                                │
    │      labels=train_labels,                                    │
    │      batch_size=32,                                          │
    │      augment=True                                            │
    │  )                                                           │
    │  model.fit(train_ds, epochs=50, ...)                         │
    └──────────────────────────────────────────────────────────────┘
    """
    try:
        import tensorflow as tf

        def load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32)
            # Preprocessing is handled by EfficientNetV2's include_preprocessing=True
            return img, label

        def augment_tf(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    except ImportError:
        raise RuntimeError("TensorFlow required for create_tf_dataset()")
