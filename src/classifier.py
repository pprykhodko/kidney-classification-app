from functools import lru_cache
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow import keras


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "kidney_classifier.keras"
CLASS_NAMES = ("Cyst", "Normal", "Stone", "Tumor")
IMAGE_SIZE = (224, 224)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Place kidney_classifier.keras in {MODEL_PATH.parent}")

    model = keras.models.load_model(MODEL_PATH, compile=False)

    if model.input_shape != (None, *IMAGE_SIZE, 3) or model.output_shape != (None, 4):
        raise ValueError("Expected a 224x224 RGB classifier with four output classes")

    return model


def load_image(source):
    try:
        data = source.read() if hasattr(source, "read") else Path(source).read_bytes()

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG", "BMP"}:
                    raise ValueError("Please upload a JPG, PNG, or BMP image")

                image.verify()

        pixels = tf.io.decode_image(data, channels=3, expand_animations=False)

        return Image.fromarray(pixels.numpy())
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        tf.errors.InvalidArgumentError
    ):
        raise ValueError(
            "The image is damaged, too large, or cannot be read"
        )


def preprocess_image(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixels = tf.image.resize(pixels, IMAGE_SIZE, method="bilinear")

    return tf.expand_dims(pixels, axis=0)


def classify_image(image):
    model = load_model()
    batch = preprocess_image(image)
    probabilities = np.asarray(model(batch, training=False))[0]
    index = int(np.argmax(probabilities))

    return {
        "class_name": CLASS_NAMES[index],
        "probability": float(probabilities[index]),
        "scores": {
            name: float(value) for name, value in zip(CLASS_NAMES, probabilities)
        }
    }
