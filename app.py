from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import inference

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    label, score, report = inference.predict_with_explainer(text)
    return jsonify({
        "label":      label,
        "confidence": float(score),
        "report":     report or {}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
