import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

#Functional API (retrieve later: error in the dense layer after flatten)
def get_model(num_words, dim_embb, max_sent_length):
    x = keras.Input(shape=(max_sent_length,))
    embedding = layers.Embedding(num_words, dim_embb, input_length=max_sent_length)(x)
    flt_1 = layers.Flatten()(embedding),
    dense_1 = layers.Dense(6, activation='relu')(embedding)
    dense_2 = layers.Dense(1, activation='sigmoid')(dense_1)

    model = keras.Model(inputs=[x],outputs=[dense_2])
    return model
