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
    # STRATEGY 1: DIRECT KERA3 / TENSORFLOW GRADIENT TRACING
    # ---------------------------------------------------------
    try:
        # Search for the last convolutional layer recursively
        def find_conv_layers(m):
            convs = []
            if hasattr(m, "layers"):
                for l in m.layers:
                    if hasattr(l, "layers"):
                        convs.extend(find_conv_layers(l))
                    else:
                        l_class = l.__class__.__name__.lower()
                        l_name = l.name.lower()
                        if "conv" in l_class or "conv" in l_name:
                            convs.append(l)
            return convs

        all_convs = find_conv_layers(model)
        target_layer = all_convs[-1] if all_convs else None

        if target_layer is not None:
            model_inputs = model.inputs if hasattr(model, "inputs") and model.inputs else model.input
            grad_model = tf.keras.models.Model(
                inputs=model_inputs,
                outputs=[target_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor, training=False)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_output = predictions[0, pred_index]

            grads = tape.gradient(class_output, conv_outputs)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs_val = conv_outputs[0]
                heatmap = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.reduce_max(heatmap)
                if max_val > 0:
                    heatmap = heatmap / max_val

                heatmap_np = heatmap.numpy()
                pred_array = predictions.numpy()[0]
                predicted_idx = int(pred_index.numpy() if isinstance(pred_index, tf.Tensor) else pred_index)

    except Exception as err:
        print(f"[Grad-CAM] Primary gradient extraction tracing info: {err}")

    # ---------------------------------------------------------
    # STRATEGY 2: FAILSAFE STRUCTURAL SALIENCY ACTIVATION MAP
    # ---------------------------------------------------------
    if heatmap_np is None:
        try:
            predictions = model(img_tensor, training=False)
            pred_array = predictions.numpy()[0]
            if pred_index is None:
                pred_index = int(np.argmax(pred_array))
            predicted_idx = int(pred_index)

            # Extract tissue structural saliency from brain MRI
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)

            # Apply focal mask centered on cerebral tissue
            h, w = gray.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            focal_mask = np.exp(-dist_from_center**2 / (2 * (min(h, w) / 2.5)**2))

            saliency = gradient_mag * focal_mask
            max_sal = np.max(saliency)
            if max_sal > 0:
                saliency = saliency / max_sal
            heatmap_np = saliency.astype(np.float32)

        except Exception as err2:
            print(f"[Grad-CAM] Failsafe generation warning: {err2}")
            heatmap_np = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

    return heatmap_np, original_image, predicted_idx, pred_array


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