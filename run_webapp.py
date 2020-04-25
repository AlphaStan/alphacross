from flask import Flask, render_template, request
import json

from src.main.environment.cross_game import CrossGame

app = Flask(__name__)
app.config.from_pyfile("webapp/config.py")
app.template_folder = app.config["TEMPLATE_FOLDER"]
app.static_folder = app.config["STATIC_FOLDER"]

game = CrossGame()

@app.route("/")
def home():
    game = CrossGame()
    return render_template('index.html',
                           title=app.config["PANE_TITLE"],
                           n_rows=game.get_n_rows(game.get_state()),
                           n_columns=game.get_n_columns(game.get_state()),
                           grid=game.get_state())

@app.route("/<column_id>", methods=['GET'])
def update_grid(column_id):
    column_id = int(column_id)
    agent_id = game.current_token_id
    game.apply_action(column_id)
    if app.config['DEBUG']:
        with open('webapp/debug/file.txt', 'a') as f:
            f.write(game.__str__())
            if game._is_winning_move(game.get_state(), column_id, agent_id):
                f.write("Congratulation player {}, you have won !\n".format(agent_id))
            if game.is_blocked():
                f.write("It's a draw !\n")
    row_id = 0
    for i, token in enumerate(game._grid[column_id][::-1]):
        if token == agent_id:
            break
        row_id += 1
    update = {'agent_id': agent_id,
              'has_won': game._is_winning_move(game.get_state(), column_id, agent_id),
              'draw': game.is_blocked(),
              'row_id': row_id,
              'col_id': column_id}
    return json.dumps(update)


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
