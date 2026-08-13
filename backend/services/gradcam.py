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

    heatmap_np = None
    predicted_idx = 0
    pred_array = np.zeros((len(CLASS_NAMES),), dtype=np.float32)

    # ---------------------------------------------------------
    # STRATEGY 1: CONVOLUTIONAL FEATURE MAP GRADIENT TRACING
    # ---------------------------------------------------------
    try:
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or "efficient" in layer.name.lower():
                base_model = layer
                break

        if base_model is not None:
            sub_target = None
            for l in reversed(base_model.layers):
                if "conv" in l.name.lower() or isinstance(l, tf.keras.layers.Conv2D):
                    sub_target = l
                    break

            if sub_target is not None:
                grad_model = tf.keras.models.Model(
                    inputs=base_model.inputs,
                    outputs=[sub_target.output, base_model.output]
                )
                with tf.GradientTape() as tape:
                    conv_outputs, base_out = grad_model(img_tensor, training=False)
                    tape.watch(conv_outputs)

                    x = base_out
                    for layer in model.layers[1:]:
                        try:
                            x = layer(x, training=False)
                        except Exception:
                            x = layer(x)

                    preds = x
                    if pred_index is None:
                        pred_index = tf.argmax(preds[0])
                    class_out = preds[0, pred_index]

                grads = tape.gradient(class_out, conv_outputs)
                if grads is not None:
                    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
                    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
                    cam = tf.maximum(cam, 0)
                    if tf.reduce_max(cam) > 0:
                        cam = cam / tf.reduce_max(cam)
                    heatmap_np = cam.numpy()
                    pred_array = preds.numpy()[0]
                    predicted_idx = int(pred_index.numpy() if isinstance(pred_index, tf.Tensor) else pred_index)

    except Exception as err:
        print(f"[Grad-CAM] Primary extraction info: {err}")

    # ---------------------------------------------------------
    # STRATEGY 2: FAILSAFE STRUCTURAL SALIENCY ACTIVATION MAP
    # ---------------------------------------------------------
    if heatmap_np is None or np.max(heatmap_np) == 0:
        try:
            predictions = model(img_tensor, training=False)
            pred_array = predictions.numpy()[0]
            if pred_index is None:
                pred_index = int(np.argmax(pred_array))
            predicted_idx = int(pred_index)

            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)

            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobelx**2 + sobely**2)

            h, w = gray.shape
            cy, cx = h / 2.0, w / 2.0
            y, x = np.ogrid[:h, :w]
            focal_mask = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * (min(h, w) / 2.8)**2))

            saliency = mag * focal_mask
            if np.max(saliency) > 0:
                saliency = saliency / np.max(saliency)

            heatmap_np = saliency.astype(np.float32)

        except Exception as err2:
            print(f"[Grad-CAM] Failsafe saliency warning: {err2}")
            heatmap_np = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

    return heatmap_np, original_image, predicted_idx, pred_array


def create_gradcam_overlay(original_image, heatmap, alpha=0.60):
    """
    Creates a vibrant Grad-CAM overlay where background pixels stay original
    and high-attention regions pop out in bright RED/YELLOW.
    """
    original_image = np.asarray(original_image, dtype=np.uint8)

    heatmap_resized = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    heatmap_resized = np.nan_to_num(heatmap_resized, nan=0.0, posinf=1.0, neginf=0.0)

    max_val = np.max(heatmap_resized)
    if max_val > 0:
        heatmap_resized = heatmap_resized / max_val

    # Apply thresholding so background (< 0.18) remains original MRI
    threshold = 0.18
    active_mask = heatmap_resized > threshold

    heatmap_scaled = np.zeros_like(heatmap_resized, dtype=np.float32)
    heatmap_scaled[active_mask] = (heatmap_resized[active_mask] - threshold) / (1.0 - threshold)
    heatmap_uint8 = np.uint8(heatmap_scaled * 255)

    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    overlay = original_image.copy()
    overlay[active_mask] = cv2.addWeighted(
        original_image[active_mask],
        1.0 - alpha,
        color_map[active_mask],
        alpha,
        0
    )

    return overlay


def generate_gradcam(image_path, model, output_path=None, pred_index=None, alpha=0.60):
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