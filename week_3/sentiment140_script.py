import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#stop words (commonly used but useless) taken from:
#https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

#HYPERPARAMETERS
VOCAB_SIZE = 10000 #(1000)
EMBEDDING_DIM = 16
MAX_LENGTH = 120 #(16)
NUM_EPOCHS = 30
NUMBER_DATA = 16
TRAIN_SIZE=0.8

#GET data
classes = []
texts = []

df = pd.read_csv("training.1600000.processed.noemoticon.csv", nrows=1600000, encoding='latin-1', names=['label' , 'ID', 'Date', 'Query', 'Name','Text'])

#@FakerPattyPattz Oh dear. Were you drinking out of the forgotten table drinks?

print(df['Text'][19])

classes = df['label'].to_list()

for text in df['Text']:
    for word in stop_words:
        text = text.lower()
        token = " " + word + " "
        text = text.replace(token, " ")
        text = text.replace("  ", " ")
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
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences_padded = pad_sequences(train_sequences,padding='post',maxlen=MAX_LENGTH)

test_sequences = tokenizer.texts_to_sequences(test_sentences) #Use the tokenizer from train set (a lot more oov since more words that are not in the train set)
test_sequences_padded = pad_sequences(test_sequences, padding='post',maxlen=MAX_LENGTH)


#LABELS
train_labels = np.array(train_labels)
train_labels = np.where(train_labels==4, 1, train_labels)
test_labels = np.array(test_labels)
test_labels = np.where(test_labels==4, 1, test_labels)

print(np.unique(test_labels))
print(np.unique(train_labels))
