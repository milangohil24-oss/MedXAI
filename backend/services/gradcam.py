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

# Global cache for the Grad-CAM feature model graph
_cached_grad_model = None


def get_last_conv_layer(model):
    """Find the last 4D feature map layer in the model."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] is not None:
                return layer
        except Exception:
            continue

    # If nested inside a submodel (e.g. EfficientNet base)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                try:
                    shape = sub_layer.output.shape
                    if len(shape) == 4 and shape[1] is not None:
                        return sub_layer
                except Exception:
                    continue

    raise ValueError("No 4D convolutional feature layer found for Grad-CAM.")


def get_grad_model(model):
    """Reuse cached Grad-CAM feature model to avoid memory spikes."""
    global _cached_grad_model
    if _cached_grad_model is not None:
        return _cached_grad_model

    target_layer = get_last_conv_layer(model)

    try:
        _cached_grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
    except Exception:
        # Fallback for nested model architectures
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is not None:
            _cached_grad_model = tf.keras.models.Model(
                inputs=base_model.inputs,
                outputs=[target_layer.output, base_model.output]
            )
        else:
            _cached_grad_model = model

    return _cached_grad_model


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"MRI image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    original_image = np.asarray(image, dtype=np.uint8)

    resized = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.asarray(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    return original_image, img_array


def make_gradcam_heatmap(image_path, model, pred_index=None):
    original_image, img_array = load_image(image_path)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    grad_model = get_grad_model(model)

    with tf.GradientTape() as tape:
        model_outputs = grad_model(img_tensor, training=False)

        if isinstance(model_outputs, (list, tuple)) and len(model_outputs) == 2:
            conv_outputs, predictions = model_outputs
        else:
            conv_outputs = model_outputs
            predictions = model(img_tensor, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        pred_index = tf.cast(pred_index, tf.int32)
        class_output = predictions[0, pred_index]

    grads = tape.gradient(class_output, conv_outputs)

    if grads is None:
        raise ValueError("Grad-CAM failure: Gradients returned None.")

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