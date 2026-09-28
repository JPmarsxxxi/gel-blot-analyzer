import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# /data on a Hugging Face Space with persistent storage; the repo's storage/ otherwise.
STORAGE_DIR = os.environ.get("GEL_STORAGE_DIR") or os.path.join(BASE_DIR, "storage")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
EXPORTS_DIR = os.path.join(STORAGE_DIR, "exports")
DB_PATH = os.path.join(STORAGE_DIR, "app.db")
DB_URI = f"sqlite:///{DB_PATH}"

MODEL_PATH = os.path.join(BASE_DIR, "app", "training", "artifacts", "band_detector.pt")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 20
# Whole-request ceiling, checked by Werkzeug before any upload is read; the
# per-file and file-count limits are enforced in the upload route.
MAX_CONTENT_LENGTH = MAX_FILES_PER_UPLOAD * MAX_FILE_BYTES + 1024 * 1024

RETENTION_DAYS = 30
UPLOAD_RATE_LIMIT = (10, 600)  # uploads per window (seconds), per client IP
REQUEST_RATE_LIMIT = (300, 60)  # requests of any kind per window, per client IP

for _d in (STORAGE_DIR, UPLOADS_DIR, EXPORTS_DIR):
    os.makedirs(_d, exist_ok=True)
