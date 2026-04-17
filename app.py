from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pymongo import MongoClient

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("emotion_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["mood_database1"]
collection = db["voice_predictions1"]

# Video links dictionary
video_links = {
    "sadness": [
        "https://youtu.be/oXv2lb99bes?si=mMJMoL66N6d0VSRJ",
        "https://youtube.com/shorts/OR7TZZznGdk?si=g67sdEc_ylHNhZpq"
    ],
    "joy": [
        "https://youtube.com/shorts/GQNW1IlvhcY?si=h4xbXHboPGB0J0QI"
    ],
    "anger": [
        "https://youtube.com/shorts/zgXWup8yNmE?si=U2AO0t6DFZo-9zmQ"
    ],
    "fear": [
        "https://youtube.com/shorts/TajAzT3CVJc?si=rZuBjbjPTfOHc07C"
    ],
    "love": [
        "https://youtu.be/aO9KoEaEkMg?si=kZ9VUjiRhM8JnYvm"
    ]
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    text = request.form["text"]

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50, padding='post')

    prediction = model.predict(padded)
    emotion_index = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([emotion_index])[0]

    collection.insert_one({
        "text": text,
        "emotion": emotion
    })

    videos = video_links.get(emotion, [])

    return render_template("index.html",
                           prediction=emotion,
                           videos=videos)

if __name__ == "__main__":
    app.run(debug=True)