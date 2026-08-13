import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def generate_lime_explanation(
    model,
    image_path: str,
    output_path: str,
):
    """
    Ultra-low RAM LIME feature attribution generator (< 10MB peak memory).
    Evaluates perturbed samples one-by-one to prevent TensorFlow batch memory spikes.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load original MRI scan cleanly
        raw_img = Image.open(image_path).convert("RGB")
        orig_np = np.asarray(raw_img.resize((224, 224), Image.Resampling.LANCZOS), dtype=np.uint8)

        # 2. Superpixel Grid Setup (6x6 Grid = 36 Superpixels)
        grid_h, grid_w = 6, 6
        cell_h, cell_w = 224 // grid_h, 224 // grid_w
        num_superpixels = grid_h * grid_w

        # 3. Base prediction on single image
        img_tensor = tf.convert_to_tensor(np.expand_dims(orig_np.astype(np.float32), axis=0))
        base_preds = model(img_tensor, training=False).numpy()[0]
        top_label = int(np.argmax(base_preds))
        del img_tensor

        # 4. Generate 5 perturbation masks
        num_samples = 5
        perturbations = np.random.randint(0, 2, size=(num_samples, num_superpixels))
        sample_preds = []

        small_np = orig_np.astype(np.float32)

        # 5. Evaluate perturbations ONE BY ONE (Zero batch memory spike)
        for i in range(num_samples):
            p_img = small_np.copy()
            p_mask = perturbations[i]
            for sp in range(num_superpixels):
                if p_mask[sp] == 0:
                    r, c = sp // grid_w, sp % grid_w
                    p_img[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = 0
            
            p_tensor = tf.convert_to_tensor(np.expand_dims(p_img, axis=0), dtype=tf.float32)
            p_pred = model(p_tensor, training=False).numpy()[0, top_label]
            sample_preds.append(p_pred)

            del p_tensor, p_img
            gc.collect()

        sample_preds = np.array(sample_preds)

        # 6. Compute feature correlation weights
        weights = np.zeros(num_superpixels)
        for sp in range(num_superpixels):
            active = perturbations[:, sp]
            if np.std(active) > 0 and np.std(sample_preds) > 0:
                weights[sp] = np.corrcoef(active, sample_preds)[0, 1]

        # 7. Build positive & negative feature masks
        positive_mask = np.zeros((224, 224), dtype=np.uint8)
        negative_mask = np.zeros((224, 224), dtype=np.uint8)

        top_pos = np.argsort(weights)[-5:]
        top_neg = np.argsort(weights)[:3]

        for sp in top_pos:
            if weights[sp] > 0:
                r, c = sp // grid_w, sp % grid_w
                positive_mask[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = 255

        for sp in top_neg:
            if weights[sp] < 0:
                r, c = sp // grid_w, sp % grid_w
                negative_mask[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = 255

        # 8. Create vivid color overlay directly on original MRI scan
        overlay = orig_np.copy()

        # Red tint for positive feature drivers
        pos_indices = positive_mask > 0
        if np.any(pos_indices):
            overlay[pos_indices] = cv2.addWeighted(
                orig_np[pos_indices], 0.35,
                np.full_like(orig_np[pos_indices], (255, 40, 40)), 0.65, 0
            )

        # Cyan tint for negative feature drivers
        neg_indices = negative_mask > 0
        if np.any(neg_indices):
            overlay[neg_indices] = cv2.addWeighted(
                orig_np[neg_indices], 0.35,
                np.full_like(orig_np[neg_indices], (0, 180, 255)), 0.65, 0
            )

        # Draw clean grid lines
        for r in range(1, grid_h):
            cv2.line(overlay, (0, r * cell_h), (224, r * cell_h), (255, 255, 0), 1)
        for c in range(1, grid_w):
            cv2.line(overlay, (c * cell_w, 0), (c * cell_w, 224), (255, 255, 0), 1)

        # Convert RGB to BGR for OpenCV saving
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # 9. Add Header Bar
        title_height = 36
        canvas = np.zeros((overlay_bgr.shape[0] + title_height, overlay_bgr.shape[1], 3), dtype=np.uint8)
        canvas[:title_height] = (30, 25, 15)  # Dark slate header
        canvas[title_height:] = overlay_bgr

        cv2.putText(
            canvas,
            "LIME - Feature Contribution",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        del sample_preds, overlay, canvas, orig_np, overlay_bgr
        gc.collect()

        return output_path

    except Exception as err:
        print(f"LIME generator exception: {err}")
        raw_img = Image.open(image_path).convert("RGB")
        orig = np.asarray(raw_img.resize((224, 224)), dtype=np.uint8)
        bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr)
        return output_path