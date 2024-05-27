from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the model
model = load_model('sentiment_model.h5')
print("Model loaded from 'sentiment_model.h5'")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded from 'tokenizer.pickle'")

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)
print("Label encoder loaded from 'label_encoder.pickle'")

# Assuming you know the max_length used during training
max_length = 50  # Replace this with the actual max length used during training

def predict_sentiment(comment):
    # Preprocess the input
    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=max_length)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Decode the predicted label
    sentiment = label_encoder.inverse_transform([predicted_class])[0]
    return sentiment

# Example usage
comment = "I am a boy"
sentiment = predict_sentiment(comment)
print(f'Sentiment: {sentiment}')
