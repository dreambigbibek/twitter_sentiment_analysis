import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


# importing data
df = pd.read_csv("./archive/twitter_training.csv")
df = pd.read_csv('./archive/twitter_training.csv', header=None, names=['column1', 'column2', 'output', 'input'])
df = df.dropna()
# print(df.head(10))

# Preprocess the data
X = df['input'].values
Y = df['output'].values
# print(X)


from sklearn.preprocessing import LabelEncoder
# label encoding
lblencoder= LabelEncoder()
y_encoded= lblencoder.fit_transform(Y)
num_classes = len(lblencoder.classes_)  # Determine the number of classes
# print(y)


from tensorflow.keras.preprocessing.text import Tokenizer
# vector tokenization using tensorflow library
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

# Pad the sequences(to make sequences of same length before training the model)
max_length = 100
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=6, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


import pickle
# Save the model
model.save('sentiment_model.h5')
print("Model saved as 'sentiment_model.h5'")

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved as 'tokenizer.pickle'")

# Save the label encoder
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(lblencoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Label encoder saved as 'label_encoder.pickle'")