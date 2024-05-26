import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# importing data
df = pd.read_csv("./archive/twitter_training.csv")
df = pd.read_csv('./archive/twitter_training.csv', header=None, names=['column1', 'column2', 'output', 'input'])
df = df.dropna()
# print(df.head(10))

# Preprocess the data
X = df['input'].values
Y = df['output'].values
# print(X)


# label encoding
lblencoder= LabelEncoder()
y= lblencoder.fit_transform(Y)
print(y)

# vector tokenization using tensorflow library
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

# print(sequences)

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
    Dense(1, activation='sigmoid')
])

model.compile()