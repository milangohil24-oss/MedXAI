import os
import gc
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]


# ============================================================
# FIND EFFICIENTNET BASE MODEL
# ============================================================

def get_efficientnet_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "efficientnet" in layer.name.lower():
                return layer

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer

    return model


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def get_last_conv_layer(model_or_base):
    for layer in reversed(model_or_base.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] is not None:
                return layer
        except Exception:
            continue

    raise ValueError("No suitable 4D convolutional feature layer found.")


# ============================================================
# LOAD AND PREPROCESS MRI
# ============================================================

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"MRI image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_image = np.asarray(image, dtype=np.uint8)

    resized = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.asarray(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return original_image, img_array


# ============================================================
# GRAD-CAM HEATMAP (WITH DUAL-FALLBACK GRAPH TRAVERSAL)
# ============================================================

def make_gradcam_heatmap(image_path, model, pred_index=None):
    original_image, img_array = load_image(image_path)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    target_layer = None
    base_model = get_efficientnet_base(model)

    # 1. Try finding target layer inside nested base model
    try:
        target_layer = get_last_conv_layer(base_model)
    except Exception:
        pass

    # 2. Fallback: search outer model directly
    if target_layer is None:
        target_layer = get_last_conv_layer(model)

    grads = None
    conv_outputs = None
    predictions = None

    # METHOD A: Direct Graph Trace (Outer Model)
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            pred_index = tf.cast(pred_index, tf.int32)
            class_output = predictions[0, pred_index]

        grads = tape.gradient(class_output, conv_outputs)
    except Exception:
        grads = None

    # METHOD B: Reconstruct Sequential Flow if Method A returned None
    if grads is None:
        sub_model = tf.keras.models.Model(
            inputs=base_model.inputs,
            outputs=[target_layer.output, base_model.output]
        )

        base_idx = None
        for idx, layer in enumerate(model.layers):
            if layer is base_model or layer.name == base_model.name:
                base_idx = idx
                break

        classifier_layers = model.layers[base_idx + 1:] if base_idx is not None else []

        with tf.GradientTape() as tape:
            conv_outputs, base_out = sub_model(img_tensor, training=False)
            x = base_out

            for layer in classifier_layers:
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)

            predictions = x

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            pred_index = tf.cast(pred_index, tf.int32)
            class_output = tf.gather(predictions[0], pred_index)

        grads = tape.gradient(class_output, conv_outputs)

    if grads is None:
        raise ValueError("Grad-CAM failure: Gradients could not be derived from model output.")

    # Compute Grad-CAM heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, tf.zeros_like(heatmap))

    heatmap_np = heatmap.numpy()
    pred_array = predictions.numpy()[0]
    pred_idx_int = int(pred_index.numpy() if isinstance(pred_index, tf.Tensor) else pred_index)

    return heatmap_np, original_image, pred_idx_int, pred_array


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_gradcam_overlay(original_image, heatmap, alpha=0.40):
    original_image = np.asarray(original_image)

    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
        else:
            original_image = np.clip(original_image, 0, 255).astype(np.uint8)

    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    heatmap_uint8 = np.uint8(heatmap * 255)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_gradcam_overlay(image_path, output_path, model, pred_index=None, alpha=0.40):
    heatmap, original_image, predicted_index, predictions = make_gradcam_heatmap(
        image_path, model, pred_index
    )

    overlay = create_gradcam_overlay(original_image, heatmap, alpha)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(overlay).save(output_path)

    return {
        "output_path": output_path,
        "predicted_index": predicted_index,
        "predictions": predictions,
        "heatmap_shape": heatmap.shape
    }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def generate_gradcam(image_path, model, output_path=None, pred_index=None, alpha=0.40):
    heatmap, original_image, predicted_index, predictions = make_gradcam_heatmap(
        image_path=image_path,
        model=model,
        pred_index=pred_index
    )

    overlay = create_gradcam_overlay(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha
    )

    saved_path = None
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        saved_path = output_path

    if 0 <= predicted_index < len(CLASS_NAMES):
        predicted_class = CLASS_NAMES[predicted_index]
    else:
        predicted_class = "Unknown"

    if 0 <= predicted_index < len(predictions):
        confidence = float(predictions[predicted_index])
    else:
        confidence = 0.0

    gc.collect()

    return {
        "heatmap": heatmap,
        "original_image": original_image,
        "overlay": overlay,
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100.0, 2),
        "predictions": predictions,
        "output_path": saved_path
    }