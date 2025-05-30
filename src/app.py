# src/app.py
from flask import Flask, request, render_template, jsonify
from qa import answer_from_text, answer_from_image_b64
import base64

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json
    if "text" in data and data["text"].strip():
        res = answer_from_text(data["text"])
    elif "image" in data:
        res = answer_from_image_b64(data["image"])
    else:
        return jsonify({"error": "No text or image provided"}), 400
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=True)
