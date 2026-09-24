from flask import Flask

from app.models.db import init_db, remove_session
from app.routes import bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.json.sort_keys = False

    init_db()
    app.register_blueprint(bp)

    @app.teardown_appcontext
    def _cleanup(_exc):
        remove_session()

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
