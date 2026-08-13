import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]


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

    # 1. Separate nested backbone layer from classifier layers
    base_layer = None
    classifier_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficient" in layer.name.lower():
            base_layer = layer
        elif base_layer is not None:
            classifier_layers.append(layer)

    if base_layer is None:
        base_layer = model
        classifier_layers = []

    # 2. Locate 4D convolutional feature layer inside the base backbone
    target_conv = None
    for layer in reversed(base_layer.layers):
        try:
            if len(layer.output_shape) == 4:
                target_conv = layer
                break
        except Exception:
            continue

    if target_conv is None:
        raise ValueError("No 4D convolutional feature layer found.")

    # 3. Create sub-model ONLY for base_layer to prevent Graph Disconnection
    base_grad_model = tf.keras.models.Model(
        inputs=base_layer.inputs,
        outputs=[target_conv.output, base_layer.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = base_grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        x = base_output
        for c_layer in classifier_layers:
            try:
                x = c_layer(x, training=False)
            except Exception:
                x = c_layer(x)

        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        pred_index = tf.cast(pred_index, tf.int32)
        class_output = predictions[0, pred_index]

    grads = tape.gradient(class_output, conv_outputs)

    if grads is None:
        raise ValueError("Gradients could not be derived.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_val = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, tf.zeros_like(heatmap))

    heatmap_np = heatmap.numpy()
    pred_array = predictions.numpy()[0]
    pred_idx_int = int(pred_index.numpy() if isinstance(pred_index, tf.Tensor) else pred_index)

    del base_grad_model, grads, pooled_grads
    gc.collect()

    return heatmap_np, original_image, pred_idx_int, pred_array


def create_gradcam_overlay(original_image, heatmap, alpha=0.40):
    original_image = np.asarray(original_image)
    if original_image.dtype != np.uint8:
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

    predicted_class = CLASS_NAMES[predicted_index] if 0 <= predicted_index < len(CLASS_NAMES) else "Unknown"
    confidence = float(predictions[predicted_index]) if 0 <= predicted_index < len(predictions) else 0.0

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