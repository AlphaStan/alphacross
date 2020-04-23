from flask import Flask, render_template

from src.main.environment.cross_game import CrossGame

app = Flask(__name__)
app.config.from_pyfile("webapp/config.py")
app.template_folder = app.config["TEMPLATE_FOLDER"]
app.static_folder = app.config["STATIC_FOLDER"]


@app.route("/")
def home():
    game = CrossGame()
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(0)
    return render_template('index.html',
                           title=app.config["PANE_TITLE"],
                           n_rows=game.get_n_rows(game.get_state()),
                           n_columns=game.get_n_columns(game.get_state()),
                           grid=game.get_state())


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
