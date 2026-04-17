import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

sentences = []
labels = []

with open("test.txt", "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            text, label = line.strip().split(";")
            sentences.append(text)
            labels.append(label)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=50, padding='post')

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

model = Sequential([
    Embedding(5000, 64, input_length=50),
    SimpleRNN(64),
    Dense(32, activation='relu'),
    Dense(len(set(labels)), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(padded, encoded_labels, epochs=20)

model.save("emotion_model.h5")

print("Model Trained Successfully!")