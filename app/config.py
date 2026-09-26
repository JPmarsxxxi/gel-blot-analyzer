import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
EXPORTS_DIR = os.path.join(STORAGE_DIR, "exports")
DB_PATH = os.path.join(STORAGE_DIR, "app.db")
DB_URI = f"sqlite:///{DB_PATH}"

# Overridable so experiments can benchmark a candidate model without replacing the shipped one.
MODEL_PATH = os.environ.get("GEL_MODEL_PATH") or os.path.join(BASE_DIR, "app", "training", "artifacts", "band_detector.pt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
MAX_CONTENT_LENGTH = None  # no explicit size cap per SPEC

for _d in (STORAGE_DIR, UPLOADS_DIR, EXPORTS_DIR):
    os.makedirs(_d, exist_ok=True)
