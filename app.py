from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

app = Flask(__name__)
model = load_model("plant_disease_model.h5")

IMAGE_SIZE = (224, 224)

# Replace with your actual class list from model training
class_names = [
"Tomato_healthy",
"Tomato__Tomato_mosaic_virus",
"Tomato__Tomato_YellowLeaf__Curl_Virus",
"Tomato__Target_Spot",
"Tomato_Spider_mites_Two_spotted_spider_mite",
"Tomato_Septoria_leaf_spot",
"Tomato_Leaf_Mold",
"Tomato_Late_blight",
"Tomato_Early_blight",
"Tomato_Bacterial_spot",
"Potato___healthy",
"Potato___Late_blight",
"Potato___Early_blight",
"Pepper__bell___healthy",
"Pepper__bell___Bacterial_spot"

]


fertilizer_suggestions = {
    "Tomato_healthy": "Use balanced NPK (10-10-10) fertilizer biweekly for optimal growth.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants and avoid tobacco handling. Use resistant varieties.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies with neem oil and plant virus-resistant varieties.",
    "Tomato__Target_Spot": "Use fungicides like chlorothalonil and ensure good air circulation.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use insecticidal soap or neem oil to control spider mites.",
    "Tomato_Septoria_leaf_spot": "Apply copper-based fungicide and avoid overhead watering.",
    "Tomato_Leaf_Mold": "Use fungicides and maintain humidity control in greenhouses.",
    "Tomato_Late_blight": "Apply chlorothalonil-based fungicide and destroy infected debris.",
    "Tomato_Early_blight": "Use mancozeb or chlorothalonil and rotate crops regularly.",
    "Tomato_Bacterial_spot": "Apply copper-based bactericide and remove infected leaves.",
    "Potato___healthy": "Apply a balanced NPK fertilizer before planting and hill soil around plants.",
    "Potato___Late_blight": "Spray with systemic fungicides like metalaxyl, and avoid excess irrigation.",
    "Potato___Early_blight": "Use fungicides such as chlorothalonil and maintain soil fertility.",
    "Pepper__bell___healthy": "Use compost-enriched soil and a balanced 5-10-10 fertilizer during fruiting.",
    "Pepper__bell___Bacterial_spot": "Apply copper sprays weekly and avoid working in wet fields."
}



def preprocess_image(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, IMAGE_SIZE)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = preprocess_image(file)
    preds = model.predict(image)

    idx = np.argmax(preds)

    # ✅ Safety check to avoid index error
    if idx >= len(class_names):
        return jsonify({"error": "Invalid prediction index."})

    predicted_class = class_names[idx]
    confidence = round(float(np.max(preds)) * 100, 2)
    suggestion = fertilizer_suggestions.get(predicted_class, "No suggestion available.")

    return jsonify({
        "class": predicted_class,
        "confidence": confidence,
        "suggestion": suggestion
    })


if __name__ == "__main__":
    app.run(debug=True)