from flask import Flask
from tensorflow.keras.models import load_model

from src.main.environment.cross_game import CrossGame
from src.main.models.dqn_agent import DQNAgent, dqn_mask_loss


def create_app():
    app = Flask(__name__)
    app.config.from_pyfile("config.py")
    app.template_folder = app.config["TEMPLATE_FOLDER"]
    app.static_folder = app.config["STATIC_FOLDER"]
    app.game = CrossGame()
    app.is_ai_active = False
    app.agent = DQNAgent(app.game, encoding='3d')
    app.agent.model = load_model(app.config["MODEL_PATH"], custom_objects={'dqn_mask_loss': dqn_mask_loss})
    return app
