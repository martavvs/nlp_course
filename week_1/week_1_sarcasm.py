import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Open json as a dictionary and get 3 vectors, one for each entry key of the dict
with open('sarcasm.json') as f:
    data_dict = json.load(f)

article_link = []
headline = []
is_sarcastic = []

for entry in data_dict:
    article_link.append(entry['article_link'])
    headline.append(entry['headline'])
    is_sarcastic.append(entry['is_sarcastic'])

#out of vocabulary token
tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(headline)
#to get a dictionary with pair-key: a number key for each new word in the input sentences
word_index = tokenizer.word_index

#Transform each text in texts in a sequence of integers.
sequences = tokenizer.texts_to_sequences(headline)

#The input of a NN needs to have the same size: padding (adds 0 to have same-size vectors
#across diferent sentences)
sequences_padded = pad_sequences(sequences, padding='post')
if __name__ == '__main__':
        print(sequences_padded)
