import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tfds.disable_progress_bar()


EMBEDDING_DIM = 64
NUM_EPOCHS = 10
BUFFER_SIZE = 1000

datasets, ds_info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

#Data is already pre-processed
train = datasets['train']
test = datasets['test']

#Subword tokenization: because each token represents a part of a word (eg., 1: 'Ten'; 2:'sor')
tokenizer = ds_info.features['text'].encoder
#atrtibutes of tokenizer: subwords and vocab_size
print(tokenizer.subwords)
print(tokenizer.vocab_size)

train_batches = (
    train.shuffle(BUFFER_SIZE).padded_batch(32,
    tf.compat.v1.data.get_output_shapes(train))) #shape(batch_size, sequence_length)

test_batches = (
    test.padded_batch(32,tf.compat.v1.data.get_output_shapes(train)))


#SEQUENTIAL MODEL API (Simple but less flexible)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, EMBEDDING_DIM),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


hist = model.fit(train_batches, epochs=NUM_EPOCHS,
            validation_data=test_batches)
