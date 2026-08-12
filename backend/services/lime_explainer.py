
import os
import cv2
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


def generate_lime_explanation(
    model,
    image_path: str,
    output_path: str
):
    """
    Generate a LIME superpixel explanation image.
    """

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    img = Image.open(
        image_path
    ).convert("RGB").resize((224, 224))

    img_array = np.array(
        img,
        dtype=np.float32
    )

    explainer = lime_image.LimeImageExplainer(
        verbose=False
    )

    def predict_fn(images):
        images = np.asarray(
            images,
            dtype=np.float32
        )

        return model.predict(
            images,
            verbose=0
        )

    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=50,
        batch_size=10
    )

    predicted_label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        predicted_label,
        positive_only=False,
        num_features=8,
        hide_rest=False
    )

    temp = np.clip(
        temp,
        0,
        255
    ).astype(np.uint8)

    marked_image = mark_boundaries(
        temp / 255.0,
        mask,
        color=(1, 1, 0),
        mode="outer"
    )

    marked_image = np.clip(
        marked_image * 255,
        0,
        255
    ).astype(np.uint8)

    marked_image = cv2.cvtColor(
        marked_image,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        output_path,
        marked_image
    )

    if not success:
        raise RuntimeError(
            f"Could not save LIME image: {output_path}"
        )

    return output_path

