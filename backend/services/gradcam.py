import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm


def load_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    img = Image.open(image_path).convert("RGB")

    original = img.copy()

    img = img.resize(
        target_size,
        Image.Resampling.BILINEAR
    )

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return original, img_array


def find_nested_efficientnet(model):
    """
    Find the nested EfficientNet model inside the
    user's classification model.
    """

    print("==============================================")
    print("SEARCHING FOR NESTED EFFICIENTNET")
    print("==============================================")

    for layer in model.layers:

        print(
            "Top-level model layer:",
            layer.name,
            type(layer).__name__
        )

        if isinstance(
            layer,
            tf.keras.Model
        ):

            print(
                "Found nested model:",
                layer.name
            )

            if (
                "efficientnet" in layer.name.lower()
                or "efficient" in layer.name.lower()
            ):

                print(
                    "Using nested model:",
                    layer.name
                )

                return layer

    raise ValueError(
        "Nested EfficientNet model could not be found."
    )


def find_last_conv_layer(nested_model):
    """
    Find the deepest convolutional layer inside
    the nested EfficientNet model.

    Prefer Conv2D / DepthwiseConv2D layers.
    """

    print("==============================================")
    print("SEARCHING FOR LAST CONVOLUTIONAL LAYER")
    print("==============================================")

    candidate_layers = []

    def search_layers(model):

        for layer in model.layers:

            if isinstance(
                layer,
                (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.DepthwiseConv2D,
                    tf.keras.layers.SeparableConv2D
                )
            ):

                try:

                    output_shape = layer.output.shape

                    if len(output_shape) == 4:

                        candidate_layers.append(
                            layer
                        )

                        print(
                            "Conv candidate:",
                            layer.name,
                            output_shape
                        )

                except Exception:
                    pass

            if isinstance(
                layer,
                tf.keras.Model
            ):

                search_layers(layer)

    search_layers(nested_model)

    if not candidate_layers:

        raise ValueError(
            "Could not find an internal convolutional "
            "layer suitable for Grad-CAM."
        )

    last_conv = candidate_layers[-1]

    print(
        "Selected Grad-CAM layer:",
        last_conv.name
    )

    print(
        "Layer output shape:",
        last_conv.output.shape
    )

    print("==============================================")

    return last_conv


def apply_classifier_head(
    model,
    efficientnet_model,
    efficientnet_output
):
    """
    Rebuild the classification head after EfficientNet.

    This avoids the 'Output with path 0 is not connected
    to inputs' problem caused by directly mixing tensors
    from nested Functional models.
    """

    x = efficientnet_output

    efficientnet_found = False

    for layer in model.layers:

        if layer is efficientnet_model:

            efficientnet_found = True
            continue

        if not efficientnet_found:
            continue

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        x = layer(
            x,
            training=False
        )

    return x


def generate_gradcam(
    model,
    image_path,
    output_path
):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"MRI image not found: {image_path}"
        )

    print("==============================================")
    print("GENERATING GRAD-CAM")
    print("==============================================")

    print(
        "Input image:",
        image_path
    )

    print(
        "Output image:",
        output_path
    )

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------

    original_image, img_array = load_image(
        image_path
    )

    # --------------------------------------------------
    # FIND NESTED EFFICIENTNET
    # --------------------------------------------------

    efficientnet_model = find_nested_efficientnet(
        model
    )

    # --------------------------------------------------
    # FIND LAST CONVOLUTIONAL LAYER
    # --------------------------------------------------

    last_conv_layer = find_last_conv_layer(
        efficientnet_model
    )

    print(
        "Grad-CAM layer:",
        last_conv_layer.name
    )

    # --------------------------------------------------
    # CREATE FEATURE MODEL
    # --------------------------------------------------

    feature_model = tf.keras.Model(
        inputs=efficientnet_model.input,
        outputs=[
            last_conv_layer.output,
            efficientnet_model.output
        ]
    )

    # --------------------------------------------------
    # FORWARD PASS + GRADIENT
    # --------------------------------------------------

    with tf.GradientTape() as tape:

        # ----------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------

        x = img_array

        augmentation_layer = None

        for layer in model.layers:

            if (
                "augmentation" in layer.name.lower()
            ):

                augmentation_layer = layer
                break

        if augmentation_layer is not None:

            print(
                "Using augmentation layer:",
                augmentation_layer.name
            )

            x = augmentation_layer(
                x,
                training=False
            )

        # ----------------------------------------------
        # EFFICIENTNET
        # ----------------------------------------------

        conv_outputs, efficientnet_output = (
            feature_model(x, training=False)
        )

        # ----------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------

        predictions = apply_classifier_head(
            model,
            efficientnet_model,
            efficientnet_output
        )

        predictions = tf.convert_to_tensor(
            predictions
        )

        print(
            "Prediction tensor shape:",
            predictions.shape
        )

        # ----------------------------------------------
        # GET PREDICTED CLASS
        # ----------------------------------------------

        if len(predictions.shape) == 2:

            prediction_vector = predictions[0]

        else:

            prediction_vector = predictions

        if prediction_vector.shape[-1] == 1:

            target_score = prediction_vector[0]

            class_index = 0

        else:

            class_index = tf.argmax(
                prediction_vector
            )

            target_score = tf.gather(
                prediction_vector,
                class_index
            )

        print(
            "Predicted class index:",
            int(class_index)
        )

        print(
            "Target score:",
            float(target_score)
        )

    # --------------------------------------------------
    # CALCULATE GRADIENTS
    # --------------------------------------------------

    grads = tape.gradient(
        target_score,
        conv_outputs
    )

    if grads is None:

        raise RuntimeError(
            "Gradients are None. "
            "Grad-CAM could not be generated."
        )

    print(
        "Gradient shape:",
        grads.shape
    )

    # --------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # --------------------------------------------------
    # CREATE HEATMAP
    # --------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = (
            heatmap / max_value
        )

    heatmap = heatmap.numpy()

    print(
        "Heatmap shape:",
        heatmap.shape
    )

    print(
        "Heatmap min:",
        float(np.min(heatmap))
    )

    print(
        "Heatmap max:",
        float(np.max(heatmap))
    )

    # --------------------------------------------------
    # RESIZE HEATMAP
    # --------------------------------------------------

    original_width, original_height = (
        original_image.size
    )

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        (
            original_width,
            original_height
        ),
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap_image,
        dtype=np.float32
    )

    heatmap_array = (
        heatmap_array / 255.0
    )

    # --------------------------------------------------
    # APPLY JET COLORMAP
    # --------------------------------------------------

    colored_heatmap = cm.jet(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    heatmap_rgb = Image.fromarray(
        colored_heatmap
    ).convert("RGB")

    # --------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------

    original_rgb = (
        original_image
        .convert("RGB")
    )

    overlay = Image.blend(
        original_rgb,
        heatmap_rgb,
        alpha=0.45
    )

    # --------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    overlay.save(
        output_path,
        format="PNG"
    )

    # --------------------------------------------------
    # VALIDATE OUTPUT
    # --------------------------------------------------

    if not os.path.exists(
        output_path
    ):

        raise RuntimeError(
            "Grad-CAM image was not created."
        )

    file_size = os.path.getsize(
        output_path
    )

    if file_size < 100:

        raise RuntimeError(
            "Grad-CAM output image is empty."
        )

    print("==============================================")
    print(
        "Grad-CAM successfully created."
    )
    print(
        "Saved to:",
        output_path
    )
    print(
        "File size:",
        file_size,
        "bytes"
    )
    print("==============================================")

    return output_path