import os
import gc

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries


# ============================================================
# LIME EXPLANATION GENERATOR
# ============================================================

def generate_lime_explanation(
    model,
    image_path: str,
    output_path: str,
):
    """
    Generate a visual LIME superpixel explanation.

    The output preserves the original MRI while highlighting
    image regions that contribute positively or negatively to
    the model's predicted class.

    Parameters
    ----------
    model:
        Loaded TensorFlow/Keras classification model.

    image_path:
        Path to the original MRI image.

    output_path:
        Destination path for the generated LIME visualization.

    Returns
    -------
    str
        Path to the generated LIME image.
    """

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # LOAD MRI
    # --------------------------------------------------------

    try:
        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize(
                (224, 224),
                Image.Resampling.LANCZOS,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not load MRI image: {error}"
        )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # MODEL PREDICTION FUNCTION (NO Keras Memory Leaks)
    # --------------------------------------------------------

    def predict_fn(images):
        """
        LIME calls this function with batches of perturbed images.
        Direct callable model(...) avoids TensorFlow memory leaks.
        """
        images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        predictions = model(images_tensor, training=False).numpy()

        if predictions.ndim != 2:
            raise RuntimeError(
                "Model prediction output must be a 2D "
                "array of shape (batch, classes)."
            )

        return predictions

    # --------------------------------------------------------
    # CREATE LIME EXPLAINER
    # --------------------------------------------------------

    explainer = lime_image.LimeImageExplainer(
        verbose=False,
    )

    # --------------------------------------------------------
    # GENERATE EXPLANATION (OPTIMIZED FOR 512MB RAM)
    # --------------------------------------------------------

    try:
        explanation = explainer.explain_instance(
            image_array,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=25,  # Reduced from 100/1000 to 25 to prevent OOM
            batch_size=1,    # Process 1 perturbed image at a time
        )
    except Exception as error:
        raise RuntimeError(
            f"LIME explanation failed: {error}"
        )

    # --------------------------------------------------------
    # GET PREDICTED CLASS
    # --------------------------------------------------------

    predicted_label = (
        explanation.top_labels[0]
    )

    # --------------------------------------------------------
    # GET POSITIVE LIME FEATURES
    # --------------------------------------------------------

    try:
        positive_image, positive_mask = (
            explanation.get_image_and_mask(
                predicted_label,
                positive_only=True,
                num_features=10,
                hide_rest=False,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not extract LIME features: {error}"
        )

    # --------------------------------------------------------
    # GET ALL IMPORTANT FEATURES
    # --------------------------------------------------------

    try:
        all_features_image, all_features_mask = (
            explanation.get_image_and_mask(
                predicted_label,
                positive_only=False,
                num_features=10,
                hide_rest=False,
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not extract LIME feature mask: {error}"
        )

    # --------------------------------------------------------
    # PREPARE ORIGINAL IMAGE
    # --------------------------------------------------------

    original = np.asarray(
        image,
        dtype=np.float32,
    )

    original = np.clip(
        original,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # BUILD LIME VISUALIZATION
    # --------------------------------------------------------

    base = (
        original.astype(np.float32)
        / 255.0
    )

    # Positive contribution regions.
    positive_mask_binary = (
        positive_mask > 0
    ).astype(np.uint8)

    # All selected regions.
    all_mask_binary = (
        all_features_mask != 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # CREATE COLORED CONTRIBUTION OVERLAY
    # --------------------------------------------------------

    overlay = np.zeros_like(
        original,
        dtype=np.uint8,
    )

    overlay[
        positive_mask_binary > 0
    ] = (0, 0, 255)

    negative_mask_binary = (
        (all_features_mask < 0)
        & (positive_mask_binary == 0)
    ).astype(np.uint8)

    overlay[
        negative_mask_binary > 0
    ] = (255, 0, 0)

    # --------------------------------------------------------
    # BLEND MRI + LIME
    # --------------------------------------------------------

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR,
    )

    blended = cv2.addWeighted(
        original_bgr,
        0.68,
        overlay,
        0.32,
        0,
    )

    # --------------------------------------------------------
    # DRAW SUPERPIXEL BOUNDARIES
    # --------------------------------------------------------

    boundary_image = mark_boundaries(
        blended.astype(np.float32) / 255.0,
        all_mask_binary,
        color=(1, 1, 0),
        mode="outer",
    )

    boundary_image = np.clip(
        boundary_image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    final_image = cv2.cvtColor(
        boundary_image,
        cv2.COLOR_RGB2BGR,
    )

    # --------------------------------------------------------
    # ADD TITLE BAR
    # --------------------------------------------------------

    title_height = 38

    canvas = np.zeros(
        (
            final_image.shape[0] + title_height,
            final_image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        title_height:
    ] = final_image

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

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    success = cv2.imwrite(
        output_path,
        canvas,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not success:
        raise RuntimeError(
            f"Could not save LIME image: {output_path}"
        )

    # --------------------------------------------------------
    # VALIDATE OUTPUT
    # --------------------------------------------------------

    if not os.path.exists(output_path):
        raise RuntimeError(
            "LIME output file was not created."
        )

    file_size = os.path.getsize(output_path)

    if file_size <= 1000:
        raise RuntimeError(
            "Generated LIME image is unexpectedly small."
        )

    # --------------------------------------------------------
    # GARBAGE COLLECTION FOR LOW RAM INSTANCES
    # --------------------------------------------------------

    gc.collect()

    return output_path