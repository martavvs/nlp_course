from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import re

#stop words (commonly used but useless) taken from:
#https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

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

#Text
tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index #dictionary with number key for each new word in the input sentences

sequences = tokenizer.texts_to_sequences(texts) #Transform each text in texts in a sequence of integers
sequences_padded = pad_sequences(sequences, padding='post') #The input of a NN needs to have the same size


#classes
classes_tokenizer = Tokenizer()
classes_tokenizer.fit_on_texts(classes)
word_index = classes_tokenizer.word_index

classes_seq = classes_tokenizer.texts_to_sequences(classes)

if __name__ == '__main__':
        print(len(texts))
        print(texts[1746])
        print(word_index)
        print(sequences[1746])
