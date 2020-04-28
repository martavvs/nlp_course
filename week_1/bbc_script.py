import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import re
import numpy as np

#stop words (commonly used but useless) taken from:
#https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

#HYPERPARAMETERS
VOCAB_SIZE = 10000 #(1000)
EMBEDDING_DIM = 16
MAX_LENGTH = 120 #(16)
NUM_EPOCHS = 30
TRAIN_SIZE = 0.8

#GET data
classes = []
texts = []
with open("classes.csv", 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data) #This method returns the next input line
    for row in data:
        classes.append(row[0])
        text = row[1]
        for word in stop_words:
            text = text.lower()
            token = " " + word + " "
            text = text.replace(token, " ")
            text = text.replace("  ", " ")
            text = re.sub('[^a-zA-Z0-9_ .-]+', '', text)
        texts.append(text)

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for i in range(0, len(texts)):
    a = np.random.uniform(0,1)
    if a < TRAIN_SIZE:
        train_sentences.append(texts[i])
        train_labels.append(classes[i])
    else:
        test_sentences.append(texts[i])
        test_labels.append(classes[i])

#TEXT
tokenizer = Tokenizer(num_words=VOCAB_SIZE,oov_token='OOV')
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index #dictionary with number key for each new word in the input sentences

train_sequences = tokenizer.texts_to_sequences(train_sentences) #Transform each text in texts in a sequence of integers
train_sequences_padded = pad_sequences(train_sequences,padding='post',maxlen=MAX_LENGTH) #The input of a NN needs to have the same size

test_sequences = tokenizer.texts_to_sequences(test_sentences) #Use the tokenizer from train set (a lot more oov since more words that are not in the train set)
test_sequences_padded = pad_sequences(test_sequences, padding='post',maxlen=MAX_LENGTH)


#LABELS
labels_tokenizer = Tokenizer()
labels_tokenizer.fit_on_texts(train_labels)

train_labels_seq = labels_tokenizer.texts_to_sequences(train_labels)
test_labels_seq = labels_tokenizer.texts_to_sequences(test_labels)
train_labels_seq = np.array(train_labels_seq)
test_labels_seq = np.array(test_labels_seq)



#SEQUENTIAL MODEL API (Simple but less flexible)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax') #6 neurons since the maximum output
    #label is 5. And so your output is from label [0,6), excluding 6
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


model.fit(train_sequences_padded, train_labels_seq, epochs=NUM_EPOCHS,
            validation_data=(test_sequences_padded, test_labels_seq))

# if __name__ == '__main__':
#     print(np.unique(test_labels_seq))
#         print(len(texts))
#         print(len(classes))
#         print(len(train_sentences))
#         print(len(train_labels))
#         print(len(test_sentences))
#         print(len(test_labels))
#         #print(texts[1746])
