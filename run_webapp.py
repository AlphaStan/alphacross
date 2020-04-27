from flask import Flask, render_template
import json
from tensorflow.keras.models import load_model
import os

from src.main.environment.cross_game import CrossGame
from src.main.environment.errors import ColumnIsFullError
from src.main.models.dqn_agent import DQNAgent, dqn_mask_loss

app = Flask(__name__)
app.config.from_pyfile("webapp/config.py")
app.template_folder = app.config["TEMPLATE_FOLDER"]
app.static_folder = app.config["STATIC_FOLDER"]

game = CrossGame()
ai_active = False
agent = DQNAgent(game)
agent.model = load_model(app.config["MODEL_PATH"], custom_objects={'dqn_mask_loss': dqn_mask_loss})


@app.route("/")
def home():
    return render_template('index.html',
                           title=app.config["PANE_TITLE"],
                           n_rows=game.get_n_rows(game.get_state()),
                           n_columns=game.get_n_columns(game.get_state()),
                           grid=game.get_state())


@app.route("/<column_id>", methods=['GET'])
def update_grid(column_id):
    global ai_active
    column_id = int(column_id)
    player_id = game.current_token_id
    column_is_full = False
    has_won = False
    is_blocked = False
    try:
        game.apply_action(column_id)
        if game._is_winning_move(game.get_state(), column_id, player_id):
            has_won = True
        elif game.is_blocked():
            is_blocked = True
    except ColumnIsFullError:
        column_is_full = True
    row_id = 0
    for i, token in enumerate(game._grid[column_id][::-1]):
        if token == player_id:
            break
        row_id += 1
    update = {'player_id': player_id,
              'has_won': has_won,
              'draw': is_blocked,
              'row_id': row_id,
              'col_id': column_id,
              'column_is_full': column_is_full}
    #TODO: on page refresh ai_active stays True but AI token are not displayed, reset to False upon refresh
    if ai_active and not has_won and not column_is_full:
        agent_id = game.current_token_id
        agent_has_won = False
        agent_has_played = False
        agent_column_id = 0
        agent_row_id = 0
        while not agent_has_played:
            try:
                agent_column_id = agent.select_action(game)
                game.apply_action(agent_column_id)
                agent_has_played = True
                if game._is_winning_move(game.get_state(), agent_column_id, agent_id):
                    agent_has_won = True
                elif game.is_blocked():
                    is_blocked = True
            except ColumnIsFullError:
                pass
        for i, token in enumerate(game._grid[agent_column_id][::-1]):
            if token == agent_id:
                break
            agent_row_id += 1
        update['agent_id'] = agent_id
        update['agent_has_won'] = agent_has_won
        update['draw'] = is_blocked
        update['agent_row_id'] = agent_row_id
        update['agent_col_id'] = int(agent_column_id)  # needs conversion because numpy.int64 is not JSON serializable
    json_file = json.dumps(update)
    return json_file


@app.route("/reset", methods=['GET', 'POST'])
def reset():
    game.reset()
    return json.dumps({"result": "SUCCESS"})


@app.route("/activation", methods=['GET'])
def activate_ai():
    global ai_active
    ai_active = not ai_active
    return json.dumps({'result': 'SUCCESS'})


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
