import os
import gc

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries


# ============================================================
# LIME EXPLANATION GENERATOR (ULTRA-LIGHT FOR 512MB RAM)
# ============================================================

def generate_lime_explanation(
    model,
    image_path: str,
    output_path: str,
):
    """
    Generate a visual LIME superpixel explanation with minimal RAM footprint.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load and downscale image to 112x112 for LIME processing (75% RAM reduction)
    try:
        raw_image = Image.open(image_path).convert("RGB")
        lime_img = raw_image.resize((112, 112), Image.Resampling.LANCZOS)
        display_img = raw_image.resize((224, 224), Image.Resampling.LANCZOS)
    except Exception as error:
        raise RuntimeError(f"Could not load MRI image: {error}")

    lime_array = np.asarray(lime_img, dtype=np.float32)

    # 2. Prediction function inside LIME using direct callable model
    def predict_fn(images):
        # Resize perturbed batch back to model input size (224x224)
        resized_batch = np.array([
            cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            for img in images
        ], dtype=np.float32)

        tensor_batch = tf.convert_to_tensor(resized_batch, dtype=tf.float32)
        preds = model(tensor_batch, training=False).numpy()

        del resized_batch, tensor_batch
        return preds

    # 3. Create explainer
    explainer = lime_image.LimeImageExplainer(verbose=False)

    try:
        explanation = explainer.explain_instance(
            lime_array,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=12,   # Reduced to 12 samples to guarantee < 512MB RAM
            batch_size=1,     # Process 1 perturbed image at a time
        )
    except Exception as error:
        raise RuntimeError(f"LIME explanation failed: {error}")

    predicted_label = explanation.top_labels[0]

    # 4. Extract feature masks
    try:
        _, positive_mask = explanation.get_image_and_mask(
            predicted_label,
            positive_only=True,
            num_features=5,
            hide_rest=False,
        )
        _, all_features_mask = explanation.get_image_and_mask(
            predicted_label,
            positive_only=False,
            num_features=5,
            hide_rest=False,
        )
    except Exception as error:
        raise RuntimeError(f"Could not extract LIME masks: {error}")

    # Resize masks back to 224x224 for display
    positive_mask = cv2.resize(
        positive_mask.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST
    )
    all_features_mask = cv2.resize(
        all_features_mask.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST
    )

    original = np.asarray(display_img, dtype=np.uint8)

    # 5. Build visualization
    positive_mask_binary = (positive_mask > 0).astype(np.uint8)
    all_mask_binary = (all_features_mask != 0).astype(np.uint8)

    overlay = np.zeros_like(original, dtype=np.uint8)
    overlay[positive_mask_binary > 0] = (0, 0, 255)

    negative_mask_binary = ((all_features_mask < 0) & (positive_mask_binary == 0)).astype(np.uint8)
    overlay[negative_mask_binary > 0] = (255, 0, 0)

    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(original_bgr, 0.68, overlay, 0.32, 0)

    boundary_image = mark_boundaries(
        blended.astype(np.float32) / 255.0,
        all_mask_binary,
        color=(1, 1, 0),
        mode="outer",
    )

    boundary_image = np.clip(boundary_image * 255.0, 0, 255).astype(np.uint8)
    final_image = cv2.cvtColor(boundary_image, cv2.COLOR_RGB2BGR)

    # 6. Title Bar
    title_height = 38
    canvas = np.zeros(
        (final_image.shape[0] + title_height, final_image.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[title_height:] = final_image

    cv2.putText(
        canvas,
        "LIME - Model Feature Contribution",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 7. Save and validate
    success = cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success or not os.path.exists(output_path):
        raise RuntimeError(f"Could not save LIME image to {output_path}")

    # Cleanup memory
    del lime_array, positive_mask, all_features_mask, overlay, blended
    gc.collect()

    return output_path