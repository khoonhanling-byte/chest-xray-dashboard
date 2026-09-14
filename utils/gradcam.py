import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# FIND CNN BACKBONE
# ============================================================

def find_backbone(model):
    """
    Find VGG16, DenseNet121 or ResNet50 inside the full model.

    The data_augmentation model is ignored.
    """

    for index, layer in enumerate(model.layers):

        if not isinstance(layer, tf.keras.Model):
            continue

        if layer.name == "data_augmentation":
            continue

        # The CNN backbone should contain Conv2D layers
        has_conv = any(
            isinstance(sub_layer, tf.keras.layers.Conv2D)
            for sub_layer in layer.layers
        )

        if has_conv:
            return layer, index

    raise ValueError(
        "CNN backbone could not be found."
    )


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(backbone):
    """
    Find the last Conv2D layer inside the CNN backbone.
    """

    for layer in reversed(backbone.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    raise ValueError(
        f"No Conv2D layer found inside {backbone.name}."
    )


# ============================================================
# RUN CLASSIFIER HEAD
# ============================================================

def run_classifier_head(
    model,
    backbone_index,
    backbone_output
):
    """
    Continue forward propagation after the backbone.

    Backbone
        -> GlobalAveragePooling
        -> Dense
        -> BatchNormalization
        -> Dropout
        -> Sigmoid
    """

    x = backbone_output

    for layer in model.layers[backbone_index + 1:]:

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)

    return x


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    model_input,
    threshold=0.50
):
    """
    Generate Grad-CAM for the class predicted by the model.

    Class mapping:
        0 = Normal-like
        1 = Pneumonia-like
    """

    model_input = tf.cast(
        model_input,
        tf.float32
    )

    # --------------------------------------------------------
    # Find pretrained CNN
    # --------------------------------------------------------

    backbone, backbone_index = find_backbone(
        model
    )

    last_conv_layer = find_last_conv_layer(
        backbone
    )


    # --------------------------------------------------------
    # Feature model
    # --------------------------------------------------------

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )


    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_output, backbone_output = feature_model(
            model_input,
            training=False
        )

        # Required because Grad-CAM needs the gradient
        # with respect to this intermediate feature map.
        tape.watch(conv_output)

        prediction = run_classifier_head(
            model,
            backbone_index,
            backbone_output
        )

        pneumonia_probability = tf.reshape(
            prediction,
            [-1]
        )[0]


        # ----------------------------------------------------
        # Determine predicted class
        # ----------------------------------------------------

        if float(pneumonia_probability.numpy()) >= threshold:

            predicted_class = 1

            # Explain Pneumonia-like prediction
            class_score = pneumonia_probability

        else:

            predicted_class = 0

            # Explain Normal-like prediction
            class_score = 1.0 - pneumonia_probability


    # --------------------------------------------------------
    # Calculate gradient
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )


    # --------------------------------------------------------
    # Importance of each feature map
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    conv_output = conv_output[0]


    # Multiply feature maps by their importance
    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1
    )


    # Keep positive contribution
    heatmap = tf.maximum(
        heatmap,
        0
    )


    # --------------------------------------------------------
    # Normalise between 0 and 1
    # --------------------------------------------------------

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum.numpy()) > 0:
        heatmap = heatmap / maximum


    return {
        "heatmap": heatmap.numpy(),

        "pneumonia_probability":
            float(pneumonia_probability.numpy()),

        "predicted_class":
            predicted_class,

        "conv_layer":
            last_conv_layer.name
    }


# ============================================================
# CLASSIC GRAD-CAM OVERLAY
# ============================================================

def overlay_heatmap(
    image,
    heatmap,
    alpha=0.45
):
    """
    Classic full-colour Grad-CAM.

    Blue   = lower influence
    Green  = moderate influence
    Yellow = stronger influence
    Red    = strongest influence
    """

    image = np.asarray(image)


    # --------------------------------------------------------
    # Convert image to uint8
    # --------------------------------------------------------

    if image.dtype != np.uint8:

        if image.max() <= 1.0:
            image = image * 255.0

        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)


    # --------------------------------------------------------
    # Make sure image has three channels
    # --------------------------------------------------------

    if image.ndim == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    elif image.ndim == 3 and image.shape[-1] == 1:

        image = np.repeat(
            image,
            3,
            axis=-1
        )


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    resized_heatmap = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )


    resized_heatmap = np.clip(
        resized_heatmap,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Convert to JET colour map
    # --------------------------------------------------------

    heatmap_uint8 = np.uint8(
        resized_heatmap * 255
    )


    coloured_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )


    # OpenCV = BGR
    # Streamlit = RGB
    coloured_heatmap = cv2.cvtColor(
        coloured_heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Blend heatmap and X-ray
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        coloured_heatmap,
        alpha,
        0
    )


    return overlay


# ============================================================
# MAIN DASHBOARD FUNCTION
# ============================================================

def make_gradcam(
    model,
    model_input,
    display_image,
    threshold=0.50,
    alpha=0.45
):
    """
    Generate complete Grad-CAM result for Streamlit.
    """

    result = generate_gradcam(
        model=model,
        model_input=model_input,
        threshold=threshold
    )


    overlay = overlay_heatmap(
        image=display_image,
        heatmap=result["heatmap"],
        alpha=alpha
    )


    if result["predicted_class"] == 1:

        class_name = "Pneumonia-like"

    else:

        class_name = "Normal-like"


    return {
        "overlay": overlay,
        "heatmap": result["heatmap"],
        "class_name": class_name,
        "predicted_class": result["predicted_class"],
        "pneumonia_probability":
            result["pneumonia_probability"],
        "conv_layer": result["conv_layer"]
    }


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================
# These are included because some of the other dashboard
# pages may still use the older function names.
# ============================================================

def make_gradcam_heatmap(
    model,
    model_input,
    threshold=0.50,
    **kwargs
):
    """
    Old dashboard compatibility function.
    """

    result = generate_gradcam(
        model=model,
        model_input=model_input,
        threshold=threshold
    )

    return result["heatmap"]


def create_overlay(
    image,
    heatmap,
    alpha=0.45,
    **kwargs
):
    """
    Alternative name for overlay_heatmap().
    """

    return overlay_heatmap(
        image=image,
        heatmap=heatmap,
        alpha=alpha
    )