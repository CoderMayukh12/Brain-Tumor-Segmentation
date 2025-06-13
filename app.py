import os
import io
import cv2
import base64
import numpy as np
import tensorflow as tf
import scipy.ndimage
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Disable GPU to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(_name_)
CORS(app)

# Model path
MODEL_PATH = "segmentation_model_new.h5"

# --- Custom Loss Functions ---
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    smooth = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tf.reduce_mean(1 - tversky_index)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.3):
    tversky = tversky_loss(y_true, y_pred, alpha, beta)
    focal_tversky = tf.math.log(tf.math.cosh(tversky * gamma))
    return focal_tversky

def combined_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=1.3, weight=0.5):
    dice = 1 - (2 * tf.reduce_sum(y_true * y_pred) + 1e-6) / \
           (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
    tversky = focal_tversky_loss(y_true, y_pred, alpha, beta, gamma)
    return weight * dice + (1 - weight) * tversky

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss=combined_loss)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Post-process segmentation mask
def postprocess_mask(mask, threshold=0.15, kernel_size=3):
    mask = mask > threshold
    mask = mask.squeeze()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = scipy.ndimage.binary_dilation(mask, structure=kernel)
    mask = scipy.ndimage.binary_erosion(mask, structure=kernel)
    return mask.astype(np.float32)

# Generate heatmap overlay and determine severity
def generate_heatmap(mask, image):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_uint8 = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    tumor_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    tumor_ratio = tumor_pixels / total_pixels

    if tumor_ratio < 0.1:
        severity = "Mild"
    elif 0.1 <= tumor_ratio < 0.3:
        severity = "Moderate"
    else:
        severity = "Severe"

    return overlay, severity

# Helper to encode numpy array image to base64 string
def encode_array_to_base64(image_array, is_rgb=True):
    if not is_rgb:
        image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("L")
    else:
        image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# API Endpoint
@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_file = request.files['image']
        image_pil = Image.open(image_file).convert('RGB')
        image_np = np.array(image_pil)

        image_resized = tf.image.resize(image_np, (128, 128))
        input_tensor = np.expand_dims(image_resized, axis=0) / 255.0

        prediction = model.predict(input_tensor)
        mask = postprocess_mask(prediction[0])

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        height, width = image_np.shape[:2]
        mask_resized = tf.image.resize(mask, (height, width)).numpy().squeeze()

        heatmap_overlay, severity = generate_heatmap(mask_resized, image_np)

        return jsonify({
            'severity': severity,
            'mask_image': encode_array_to_base64(mask_resized, is_rgb=False),
            'heatmap_image': encode_array_to_base64(heatmap_overlay, is_rgb=True)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    app.run(debug=True)