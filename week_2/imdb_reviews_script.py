import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tfds.disable_progress_bar()


from model_imdb_reviews import get_model

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16 #nb dimension of the vector representing the word encoding
MAX_LENGTH = 120
NUM_EPOCHS = 10

datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_size = info.splits['train'].num_examples

train = datasets['train']
test = datasets['test']

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

#copy from tf arrays to arrays of strings
for sentence, label in train:
    train_sentences.append(str(sentence.numpy()))
    train_labels.append(label.numpy())

for sentence, label in test:
    test_sentences.append(str(sentence.numpy()))
    test_labels.append(label.numpy())

#LABELS: need np arrays and not just lists
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

#SENTENCES
tokenizer = Tokenizer(num_words = VOCAB_SIZE ,oov_token='OOV')
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index #dictionary with number key for each new word in the input sentences

train_sequences = tokenizer.texts_to_sequences(train_sentences) #Transform each text in texts in a sequence of integers
train_sequences_padded = pad_sequences(train_sequences,maxlen=MAX_LENGTH,truncating='post') #The input of a NN needs to have the same size

test_sequences = tokenizer.texts_to_sequences(test_sentences) #Use the tokenizer from train set (a lot more oov since more words that are not in the train set)
test_sequences_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH)


#SEQUENTIAL MODEL API (Simple but less flexible)
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(6, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


hist = model.fit(train_sequences_padded, train_labels, epochs=NUM_EPOCHS,
            validation_data=(test_sequences_padded, test_labels))

print(hist.history['accuracy'])
