import math
import os

from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import MAX_CONTENT_LENGTH, MAX_FILE_BYTES, MAX_FILES_PER_UPLOAD, REQUEST_RATE_LIMIT, UPLOAD_RATE_LIMIT
from app.models.db import init_db, remove_session
from app.ratelimit import RateLimiter
from app.routes import bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    # Spaces (and most hosts) sit behind one reverse proxy; trust its X-Forwarded-For.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    init_db()
    app.register_blueprint(bp)
    limiter = RateLimiter()

    @app.before_request
    def _rate_limit():
        if not app.config.get("RATE_LIMITS", True):
            return None
        ip = request.remote_addr or "unknown"
        retry = limiter.hit(f"all:{ip}", *REQUEST_RATE_LIMIT)
        if retry is None and request.method == "POST" and request.path == "/api/projects":
            retry = limiter.hit(f"upload:{ip}", *UPLOAD_RATE_LIMIT)
        if retry is not None:
            wait = math.ceil(retry)
            response = jsonify({"error": f"Too many requests. Try again in {wait} second{'s' if wait != 1 else ''}."})
            response.status_code = 429
            response.headers["Retry-After"] = str(wait)
            return response
        return None

    @app.errorhandler(413)
    def _too_large(_exc):
        return jsonify({"error": f"Upload too large. Send at most {MAX_FILES_PER_UPLOAD} images of up to {MAX_FILE_BYTES // (1024 * 1024)} MB each."}), 413

    @app.teardown_appcontext
    def _cleanup(_exc):
        remove_session()

    return app


if __name__ == "__main__":
    # Development only; deploy with gunicorn (app/wsgi.py). The Werkzeug
    # debugger allows remote code execution, so it's opt-in via GEL_DEBUG=1.
    create_app().run(debug=os.environ.get("GEL_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
