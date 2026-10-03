"""Flask web app: open camera / upload photo -> get waste category.

Run:  python app.py   then open http://127.0.0.1:5000
(On a phone, open http://<your-pc-ip>:5000 on the same Wi-Fi; camera access on
non-localhost needs HTTPS, so the upload-from-camera button is the fallback.)
"""
import io
from flask import Flask, jsonify, render_template, request

from classifier import classify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image received"), 400
    try:
        return jsonify(classify(io.BytesIO(f.read())))
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except Exception as e:  # corrupt image etc.
        return jsonify(error=f"Could not process image: {e}"), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
