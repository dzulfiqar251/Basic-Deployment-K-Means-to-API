from flask import Flask, render_template, request
import pandas as pd
from jcopml.utils import load_model

app = Flask(__name__)
model = load_model("model/km.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("upload.html")
    elif request.method == "POST":
        csv_file = request.files.get("file")
        X_test = pd.read_csv(csv_file, usecols=["X","Y"])
        X_test["klabel"] = model.predict(X_test)
        return X_test.to_html()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")