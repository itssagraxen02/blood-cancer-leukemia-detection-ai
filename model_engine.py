"""
model_engine.py
───────────────
Core model logic:
  - EfficientNetV2-B0 architecture builder (Transfer Learning)
  - Grad-CAM implementation (Explainable AI / XAI)
  - Demo mode with mock weights (fully functional for UI demo)
  - Clear placeholders for linking trained .h5 or .pth files
"""

import numpy as np
import cv2
from PIL import Image
import time
import os

# ══════════════════════════════════════════════════════════════════════════════
# TENSORFLOW IMPORTS (with graceful fallback for environments without GPU)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetV2B0
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not found. Running in FULL DEMO mode.")


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE         = 224
NUM_CLASSES      = 4
CLASS_NAMES      = ["Healthy", "ALL", "AML", "CML"]
DEMO_MODEL_PATH  = "models/leukemia_model.h5"   # ← LINK YOUR .h5 HERE
GRADCAM_LAYER    = "block6a_expand_activation"   # Last conv block of EfficientNetV2-B0


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_efficientnetv2_model(num_classes: int = NUM_CLASSES,
                                input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3),
                                freeze_base: bool = True) -> "keras.Model":
    """
    Build EfficientNetV2-B0 with custom classification head.
    
    Architecture:
      Input → EfficientNetV2-B0 (ImageNet) → GlobalAvgPool
           → Dense(256, GELU) → Dropout(0.4)
           → Dense(128, GELU) → Dropout(0.3)
           → Dense(4, Softmax)
    
    Args:
        num_classes  : Number of output classes (4: Healthy/ALL/AML/CML)
        input_shape  : Tuple (H, W, C)
        freeze_base  : If True, freeze backbone for initial fine-tuning phase
    
    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required to build model.")

    inputs = keras.Input(shape=input_shape, name="input_layer")

    # ── Backbone: EfficientNetV2-B0 ──────────────────────────────────────────
    # include_top=False removes the ImageNet classification head
    base_model = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",        # Transfer Learning from ImageNet
        input_tensor=inputs,
        include_preprocessing=True # Built-in normalization
    )
    base_model.trainable = not freeze_base

    # ── Custom Classification Head ───────────────────────────────────────────
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense Block 1
    x = layers.Dense(256, name="dense_1")(x)
    x = layers.Activation("gelu", name="gelu_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)

    # Dense Block 2
    x = layers.Dense(128, name="dense_2")(x)
    x = layers.Activation("gelu", name="gelu_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LIAS_EfficientNetV2B0")

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model


def build_demo_model():
    """
    Lightweight mock model that mimics the real architecture output.
    Used when TensorFlow is unavailable or for rapid UI demos.
    Returns a callable Python object that produces realistic-looking outputs.
    """
    class MockModel:
        def __init__(self):
            self.trainable_params = "7.2M"
            self.name = "LIAS_EfficientNetV2B0_DEMO"
            self._is_demo = True

        def predict(self, x):
            """Produces deterministic demo predictions based on image statistics."""
            np.random.seed(int(np.mean(x) * 1000) % 2**32)
            # Simulate softmax output with one dominant class
            raw = np.random.dirichlet(alpha=[3, 1, 1, 1])
            return raw.reshape(1, -1)

        def __call__(self, x, training=False):
            return self.predict(x)

    return MockModel()


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL LOADING — REAL vs DEMO
# ══════════════════════════════════════════════════════════════════════════════

def build_or_load_model(model_path: str = DEMO_MODEL_PATH):
    """
    Smart loader: tries real model first, gracefully falls back to demo.
    
    ┌─────────────────────────────────────────────────────────────┐
    │  TO USE YOUR TRAINED MODEL:                                 │
    │  1. Train using train.py (see training script)              │
    │  2. Save: model.save('models/leukemia_model.h5')            │
    │  3. Set DEMO_MODEL_PATH = 'models/leukemia_model.h5'        │
    │  4. Restart the Streamlit app                               │
    │                                                             │
    │  FOR PYTORCH (.pth) MODELS:                                 │
    │  → See the PyTorchAdapter class at the bottom of this file  │
    └─────────────────────────────────────────────────────────────┘
    """
    # Try loading real model
    if TF_AVAILABLE and os.path.exists(model_path):
        try:
            print(f"[LIAS] Loading trained model from: {model_path}")
            model = tf.keras.models.load_model(model_path)
            print(f"[LIAS] ✓ Trained model loaded successfully")
            model._is_demo = False
            model.trainable_params = f"{model.count_params()/1e6:.1f}M"
            return model
        except Exception as e:
            print(f"[LIAS] Failed to load model: {e}. Falling back to demo.")

    # Try building with TF (builds architecture with random weights)
    if TF_AVAILABLE:
        try:
            print("[LIAS] Building EfficientNetV2-B0 with ImageNet weights (demo mode)...")
            model = build_efficientnetv2_model()
            model._is_demo = True
            model.trainable_params = f"{model.count_params()/1e6:.1f}M"
            print(f"[LIAS] ✓ Architecture built — {model.trainable_params} parameters")
            return model
        except Exception as e:
            print(f"[LIAS] TF model build failed: {e}. Using mock model.")

    # Full fallback: mock model (no TF dependency)
    print("[LIAS] Running in FULL MOCK mode (no TensorFlow required)")
    return build_demo_model()


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAD-CAM IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Paper: "Grad-CAM: Visual Explanations from Deep Networks via 
            Gradient-based Localization" (Selvaraju et al., 2017)
    
    Process:
      1. Forward pass through the model
      2. Get the output of the last convolutional layer
      3. Compute gradients of the predicted class w.r.t that layer
      4. Pool gradients spatially (global average pooling)
      5. Weight feature maps by pooled gradients
      6. Apply ReLU and normalize → heatmap
      7. Bilinear upsample to input resolution
    """

    def __init__(self, model, layer_name: str = GRADCAM_LAYER):
        self.model = model
        self.layer_name = layer_name
        self._grad_model = None

        if TF_AVAILABLE and hasattr(model, 'layers'):
            self._build_grad_model()

    def _build_grad_model(self):
        """Create a sub-model that outputs both feature maps and predictions."""
        try:
            # Find the target layer
            target_layer = self.model.get_layer(self.layer_name)
            self._grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception:
            # Try alternative layers if named layer not found
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:  # Conv layer has 4D output (B,H,W,C)
                    self._grad_model = tf.keras.models.Model(
                        inputs=self.model.inputs,
                        outputs=[layer.output, self.model.output]
                    )
                    break

    def compute(self, img_array: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given image.
        
        Args:
            img_array : Preprocessed image (H, W, 3) uint8 or float
            class_idx : Target class index. If None, uses predicted class.
        
        Returns:
            heatmap : Normalized heatmap array (H, W) in [0, 1]
        """
        if self._grad_model is None:
            return self._demo_heatmap(img_array.shape[:2])

        # Prepare input tensor
        img_tensor = tf.cast(
            tf.expand_dims(
                tf.image.resize(img_array, [IMG_SIZE, IMG_SIZE]),
                axis=0
            ),
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._grad_model(img_tensor, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, class_idx]

        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)

        # Global Average Pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + Normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize to original image size
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        return heatmap

    def _demo_heatmap(self, shape: tuple) -> np.ndarray:
        """
        Generate a realistic-looking Grad-CAM heatmap for demo mode.
        Simulates model attention on cell nuclei (center-biased with noise).
        """
        h, w = shape
        # Create multiple Gaussian blobs (simulating cell attention)
        heatmap = np.zeros((h, w), dtype=np.float32)
        np.random.seed(42)

        num_blobs = np.random.randint(3, 7)
        for _ in range(num_blobs):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            sigma = np.random.randint(min(h, w) // 8, min(h, w) // 4)
            strength = np.random.uniform(0.4, 1.0)

            y, x = np.ogrid[:h, :w]
            blob = strength * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            heatmap += blob

        heatmap = np.clip(heatmap, 0, 1)
        heatmap /= heatmap.max() + 1e-8
        return heatmap


def apply_heatmap_overlay(original_img: np.ndarray,
                           heatmap: np.ndarray,
                           alpha: float = 0.6) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.
    
    Args:
        original_img : RGB image array (H, W, 3)
        heatmap      : Normalized heatmap (H, W) in [0, 1]
        alpha        : Overlay opacity
    
    Returns:
        Blended RGB overlay image
    """
    # Apply JET colormap (blue=low attention, red=high attention)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original
    if heatmap_colored.shape[:2] != original_img.shape[:2]:
        heatmap_colored = cv2.resize(
            heatmap_colored,
            (original_img.shape[1], original_img.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Blend
    orig_float = original_img.astype(np.float32)
    heat_float = heatmap_colored.astype(np.float32)
    overlay = (1 - alpha) * orig_float + alpha * heat_float
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# 4. UNIFIED PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_gradcam(model, img_array: np.ndarray, alpha: float = 0.6):
    """
    End-to-end prediction pipeline with Grad-CAM visualization.
    
    Steps:
      1. Resize and normalize image for model input
      2. Run forward pass → get class probabilities
      3. Compute Grad-CAM heatmap for predicted class
      4. Overlay heatmap on original image
      5. Return predictions, overlay, and performance stats
    
    Args:
        model     : Keras model or MockModel
        img_array : Raw input image (H, W, 3) uint8
        alpha     : Grad-CAM overlay opacity
    
    Returns:
        Tuple(predictions_array, gradcam_pil_image, stats_dict)
    """
    t_start = time.time()

    # ── Preprocess for model input ────────────────────────────────────────────
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_float   = img_resized.astype(np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────
    t_infer = time.time()
    if TF_AVAILABLE and hasattr(model, 'predict'):
        img_batch   = np.expand_dims(img_float, axis=0)
        predictions = model.predict(img_batch, verbose=0)[0]
    else:
        predictions = model.predict(img_float)[0]
    infer_ms = int((time.time() - t_infer) * 1000)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam = GradCAM(model, layer_name=GRADCAM_LAYER)
    top_class_idx = int(np.argmax(predictions))
    heatmap = gradcam.compute(img_float, class_idx=top_class_idx)

    # ── Overlay ───────────────────────────────────────────────────────────────
    overlay = apply_heatmap_overlay(img_array, heatmap, alpha=alpha)
    overlay_pil = Image.fromarray(overlay)

    total_ms = int((time.time() - t_start) * 1000)

    # ── Collect Stats ─────────────────────────────────────────────────────────
    model_params = getattr(model, 'trainable_params', '7.2M')
    stats = {
        "inference_ms": infer_ms,
        "preproc_ms"  : total_ms - infer_ms,
        "total_ms"    : total_ms,
        "model_params": model_params,
        "is_demo"     : getattr(model, '_is_demo', True),
        "img_size"    : f"{img_array.shape[1]}×{img_array.shape[0]}"
    }

    return predictions, overlay_pil, stats


# ══════════════════════════════════════════════════════════════════════════════
# 5. PYTORCH ADAPTER (PLACEHOLDER)
# ══════════════════════════════════════════════════════════════════════════════

class PyTorchAdapter:
    """
    Adapter to use a PyTorch (.pth) model with this system.
    
    ┌────────────────────────────────────────────────────────────────┐
    │  HOW TO USE YOUR PYTORCH MODEL:                               │
    │                                                               │
    │  1. Install: pip install torch torchvision                    │
    │  2. Uncomment the import below                                │
    │  3. Pass your .pth path to __init__                           │
    │  4. In app.py → load_model(), replace:                        │
    │       model = build_or_load_model()                           │
    │     With:                                                     │
    │       model = PyTorchAdapter('models/my_model.pth')           │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, pth_path: str):
        # import torch
        # import torchvision.models as tv_models
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # # Option A: Load a full saved model
        # self.model = torch.load(pth_path, map_location=self.device)
        #
        # # Option B: Load state_dict into architecture
        # self.model = tv_models.efficientnet_v2_s(pretrained=False)
        # self.model.classifier[1] = torch.nn.Linear(1280, NUM_CLASSES)
        # self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
        #
        # self.model.eval()
        self._is_demo = False
        self.trainable_params = "~7M"
        raise NotImplementedError(
            "Uncomment the PyTorch code in PyTorchAdapter.__init__() "
            "and install torch/torchvision to use .pth models."
        )

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """
        Run inference with PyTorch model.
        Returns softmax probabilities as numpy array.
        """
        # import torch
        # import torch.nn.functional as F
        # from torchvision import transforms
        #
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        # tensor = transform(img_array).unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     logits = self.model(tensor)
        #     probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        # return probs
        pass
