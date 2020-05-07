import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
import numpy as np
import re

data = open("sonnets.txt").read()
sentences = data.lower().split('\n')

print(sentences[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

num_words = len(tokenizer.word_index) + 1

print(num_words)

sequences = tokenizer.texts_to_sequences(sentences)

x_sequences = []
y_labels = []
for token in sequences:
    print(token)
    if len(token)>0:
        x_sequences.append(token[:-1])
        y_labels.append(token[-1])

num_sentences = len(x_sequences)

#one-hot encoded the y_labels
y_labels = tf.keras.utils.to_categorical(y_labels, num_classes=num_words)
print(y_labels[0])

MAX_LEN = max([len(sentence) for sentence in x_sequences])
print(MAX_LEN)

x_sequences = np.array(pad_sequences(x_sequences,padding= 'pre' ,maxlen=MAX_LEN))


#MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, 64, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences = True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(num_words/2,kernel_regularizer=regularizers.l2(0.01),activation= 'relu'),
    tf.keras.layers.Dense(num_words, activation= 'softmax' )
])
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss= 'categorical_crossentropy' ,optimizer= opt,metrics=[ 'accuracy' ])
model.summary()

NUM_EPOCHS = 100
hist = model.fit(x_sequences, y_labels, epochs=NUM_EPOCHS)

#Prediction

test_sentence =  '"Help me Obi Wan Kenobi, youre my only hope'
next_words_pred = 100

for i in range(next_words_pred):
    test_tokenized = tokenizer.texts_to_sequences(test_sentence)
    test_tokenized = np.array(pad_sequences(test_tokenized,padding= 'pre' ,maxlen=MAX_LEN))
    prediction = model.predict_classes(test_tokenized)
    pred_sentence = test_sentence
    for j in range(len(prediction)):
        for word, i in tokenizer.word_index.items():
            if prediction[j]==i:
                pred_sentence = pred_sentence +' ' + word
print(pred_sentence)
print(prediction)
