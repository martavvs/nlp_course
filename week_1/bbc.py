import json
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

#create a .csv file from the bbc/ folder that can be found here:
#http://mlg.ucd.ie/datasets/bbc.html
def prepare_data(dir):
    with open('classes.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('classes', 'text'))
    subdirs = [subdir for subdir in os.listdir(dir) if not subdir.endswith(".TXT")]
    for subdir in subdirs:
        print(subdir)
        dir_files = dir + subdir +'/'
        for filename in os.listdir(dir_files):
            try:
                with open(dir_files + filename, 'r') as f:
                    # classes[str(counter)].append(f.readlines())
                    lines = f.read().splitlines()
                    with open('classes.csv', 'a+') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow((str(subdir), lines))
            except:
                'Error'

#
# article_link = []
# headline = []
# is_sarcastic = []
#
# for entry in data_dict:
#     article_link.append(entry['article_link'])
#     headline.append(entry['headline'])
#     is_sarcastic.append(entry['is_sarcastic'])

# #out of vocabulary token
# tokenizer = Tokenizer(oov_token='OOV')
# tokenizer.fit_on_texts(headline)
# #to get a dictionary with pair-key: a number key or each new word in the input sentences
# word_index = tokenizer.word_index
#
# #Transform each text in texts in a sequence of integers.
# sequences = tokenizer.texts_to_sequences(headline)
#
# #The input of a NN needs to have the same size: padding (adds 0 to have same-size vectors
# #across diferent sentences)
# sequences_padded = pad_sequences(sequences, padding='post')
if __name__ == '__main__':
        dir = 'bbc/'
        prepare_data(dir)
