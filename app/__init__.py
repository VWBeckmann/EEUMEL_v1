import os
import logging
from flask import Flask
from dotenv import load_dotenv

def create_app():
    load_dotenv()
    
    app = Flask(__name__)

    from .routes import bp as main_bp
    app.register_blueprint(main_bp)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    return app
