from flask import Flask, render_template, current_app
import json

from src.main.environment.errors import ColumnIsFullError
from webapp.factory import create_app


app = create_app()


@app.route("/")
def home():
    return render_template('index.html',
                           title=app.config["PANE_TITLE"],
                           n_rows=current_app.game.get_n_rows(current_app.game.get_state()),
                           n_columns=current_app.game.get_n_columns(current_app.game.get_state()),
                           grid=current_app.game.get_state())


@app.route("/<column_id>/<is_ai_active>", methods=['GET'])
def update_grid(column_id, is_ai_active):
    column_id = int(column_id)
    is_ai_active = False if is_ai_active=='false' else True
    player_id = current_app.game.current_token_id
    column_is_full = False
    has_won = False
    is_blocked = False
    try:
        current_app.game.apply_action(column_id)
        if current_app.game._is_winning_move(current_app.game.get_state(), column_id, player_id):
            has_won = True
        elif current_app.game.is_blocked():
            is_blocked = True
    except ColumnIsFullError:
        column_is_full = True
    row_id = 0
    for i, token in enumerate(current_app.game._grid[column_id][::-1]):
        if token == player_id:
            break
        row_id += 1
    update = {'player_id': player_id,
              'has_won': has_won,
              'draw': is_blocked,
              'row_id': row_id,
              'col_id': column_id,
              'column_is_full': column_is_full}
    # TODO: on page refresh ai_active stays True but AI token are not displayed, reset to False upon refresh
    if is_ai_active and not has_won and not column_is_full:
        agent_id = current_app.game.current_token_id
        agent_has_won = False
        agent_has_played = False
        agent_column_id = 0
        agent_row_id = 0
        while not agent_has_played:
            try:
                agent_column_id = current_app.agent.select_action(current_app.game)
                current_app.game.apply_action(agent_column_id)
                agent_has_played = True
                if current_app.game._is_winning_move(current_app.game.get_state(), agent_column_id, agent_id):
                    agent_has_won = True
                elif current_app.game.is_blocked():
                    is_blocked = True
            except ColumnIsFullError:
                pass
        for i, token in enumerate(current_app.game._grid[agent_column_id][::-1]):
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
    current_app.game.reset()
    return json.dumps({"result": "SUCCESS"})


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
