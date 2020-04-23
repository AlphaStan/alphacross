from flask import Flask, render_template


app = Flask(__name__)
app.config.from_pyfile("webapp/config.py")
app.template_folder = app.config["TEMPLATE_FOLDER"]
app.static_folder = app.config["STATIC_FOLDER"]


@app.route("/")
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
