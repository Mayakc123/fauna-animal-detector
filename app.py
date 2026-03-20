from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load model lazily to avoid startup timeout
model = None

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        model = MobileNetV2(weights='imagenet')
        print("Model loaded!")
    return model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB").resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        m = get_model()
        predictions = m.predict(img_array)
        results = decode_predictions(predictions, top=3)[0]

        top_results = []
        for (id, label, confidence) in results:
            top_results.append({
                "label": label.replace("_", " ").title(),
                "confidence": round(float(confidence) * 100, 1)
            })

        return jsonify({"results": top_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)