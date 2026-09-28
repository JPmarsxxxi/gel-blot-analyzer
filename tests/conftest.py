import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
from PIL import Image

from app.server import create_app
from app.training.synth_data import generate_sample


@pytest.fixture()
def client():
    app = create_app()
    app.testing = True
    app.config["RATE_LIMITS"] = False
    with app.test_client() as c:
        yield c


def synth_png_bytes(seed: int = 1) -> bytes:
    sample = generate_sample(seed=seed)
    buf = io.BytesIO()
    Image.fromarray(sample.image).save(buf, format="PNG")
    return buf.getvalue()
