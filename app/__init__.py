"""MedBot — AI health-care chatbot web application."""

from __future__ import annotations

import logging
import os

from flask import Flask

__version__ = "2.0.0"


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=32 * 1024,
        PRELOAD_MODELS=os.environ.get("PRELOAD_MODELS", "1") == "1",
    )
    if config:
        app.config.update(config)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

    from .routes import api, pages

    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")

    @app.context_processor
    def inject_globals():
        return {"app_version": __version__}

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return resp

    if app.config["PRELOAD_MODELS"]:
        from .services import get_bot, get_heart_model

        get_bot()
        get_heart_model()

    return app
