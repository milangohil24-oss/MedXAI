
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "alzheimer_efficientnetb0.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)


# ============================================================
# CLASS NAMES
# ============================================================

if os.path.exists(CLASS_NAMES_PATH):

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as f:

        CLASS_NAMES = json.load(f)

else:

    CLASS_NAMES = [
        "Non Demented",
        "Very Mild Demented",
        "Mild Demented",
        "Moderate Demented"
    ]


# ============================================================
# MODEL CACHE
# ============================================================

_model = None


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():

    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                "Trained EfficientNetB0 model not found at: "
                f"{MODEL_PATH}"
            )

        print(
            f"Loading model from: {MODEL_PATH}"
        )

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print(
            "Model loaded successfully."
        )

        print(
            "Input shape:",
            _model.input_shape
        )

        print(
            "Output shape:",
            _model.output_shape
        )

    return _model


# ============================================================
# PREPROCESS MRI
# ============================================================

def preprocess_mri(
    image_path: str,
    target_size=(224, 224)
):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    img = Image.open(
        image_path
    ).convert("RGB")

    img = img.resize(
        target_size,
        Image.Resampling.BILINEAR
    )

    img_array = np.asarray(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# PREDICT MRI
# ============================================================

def predict_mri(
    image_path: str
):

    model = get_model()

    processed_img = preprocess_mri(
        image_path
    )

    predictions = model.predict(
        processed_img,
        verbose=0
    )

    preds = np.asarray(
        predictions,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # REMOVE BATCH DIMENSION
    # --------------------------------------------------------

    if preds.ndim == 2:

        preds = preds[0]

    # --------------------------------------------------------
    # BINARY OUTPUT SAFETY
    # --------------------------------------------------------

    if len(preds) == 1:

        probability = float(
            preds[0]
        )

        probability = np.clip(
            probability,
            0.0,
            1.0
        )

        preds = np.array(
            [
                1.0 - probability,
                probability
            ],
            dtype=np.float32
        )

    # --------------------------------------------------------
    # SOFTMAX IF MODEL RETURNS LOGITS
    # --------------------------------------------------------

    if (
        np.min(preds) < 0
        or
        np.max(preds) > 1
        or
        not np.isclose(
            np.sum(preds),
            1.0,
            atol=1e-3
        )
    ):

        exp_preds = np.exp(
            preds - np.max(preds)
        )

        preds = (
            exp_preds /
            np.sum(exp_preds)
        )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    preds = np.clip(
        preds,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    top_idx = int(
        np.argmax(preds)
    )

    if top_idx >= len(CLASS_NAMES):

        raise ValueError(
            "Model output classes do not match "
            "class_names.json"
        )

    predicted_class = CLASS_NAMES[
        top_idx
    ]

    confidence = float(
        preds[top_idx]
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        if index < len(preds):

            probabilities[class_name] = float(
                preds[index]
            )

        else:

            probabilities[class_name] = 0.0

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "prediction": predicted_class,

        "confidence": confidence,

        "confidence_percentage": round(
            confidence * 100,
            2
        ),

        "probabilities": probabilities
    }

