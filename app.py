from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)

# Load the new model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
print("Multi-animal model loaded!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    
    # Prepare image (MobileNetV2 needs 224x224)
    img = Image.open(file.stream).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    results = decode_predictions(predictions, top=3)[0]
    
    # Format results
    top_results = []
    for (id, label, confidence) in results:
        top_results.append({
            "label": label.replace("_", " ").title(),
            "confidence": round(float(confidence) * 100, 1)
        })
    
    return jsonify({"results": top_results})

if __name__ == "__main__":
    app.run(debug=True)