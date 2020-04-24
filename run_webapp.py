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
    with open('webapp/debug/file.txt', 'a') as f:
        f.write(game.__str__())
        if game._is_winning_move(game.get_state(), column_id, agent_id):
            f.write("Congratulation player {}, you have won !\n".format(agent_id))
        if game.is_blocked():
            f.write("It's a draw !\n")
    return json.dumps({"TODO": "data to return to js, probably the updated grid so it dynamically updates the display"})



if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
