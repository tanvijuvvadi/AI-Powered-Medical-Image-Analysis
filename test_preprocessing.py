import numpy as np
from src.preprocessing import apply_clahe, to_model_input


def test_model_input_shape_and_range():
    gray = (np.random.rand(300, 400) * 255).astype("uint8")
    x = to_model_input(gray)
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32 and 0 <= x.min() and x.max() <= 255


def test_clahe_keeps_shape():
    gray = (np.random.rand(128, 128) * 255).astype("uint8")
    assert apply_clahe(gray).shape == gray.shape
