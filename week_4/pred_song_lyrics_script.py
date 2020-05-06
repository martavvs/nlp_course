import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

tokenizer = Tokenizer()

data =  'In the town of Athy one Jeremy Lanigan \n battered away till he hadnt a pound. \n His father he died and made him a man again \n his father he died and made him a man again \n left a farm with ten acres of ground \n he gave a grand party for friends a relations \n who did not forget him when come to the will \n and if youll but listen Ill make youre eyes glisten \n of rows and ructions at Lanigans Ball six long months I spent in Dublin \n six long months doing nothin at all \n six long months I spent in Dublin \n learning to dance for Lanigans Ball \n I stepped out I stepped in again \n I stepped out I stepped in again \n I stepped out I stepped in again \n learning to dance for Lanigans Ball \n Myself to be sure got free invitaions \n for all the nice boys and girls I did ask \n in less than 10 minutes the friends and relations \n were dancing as merry as bee  round a cask \n There was lashing of punch and wine for the ladies \n potatoes and cakes there was bacon a tay \n there were the O Shaughnessys, Murphys, Walshes, O Gradys \n courtin  the girls and dancing away \n they were doing all kinds of nonsensical polkas \n all  round the room in a whirly gig \n but Julia and I soon banished their nonsense \n and tipped them a twist of a real Irish jig \n Oh how that girl got mad on me \n and danced till you d think the ceilings would fall \n for I spent three weeks at Brook s academy\n learning to dance for Lanigan s Ball \n The boys were all merry the girls were all hearty \n dancing away in couples and groups \n till an accident happened young Terrance McCarthy \n put his right leg through Miss Finerty s hoops \n The creature she fainted and cried  melia murder \n cried for her brothers and gathered them all \n Carmody swore that he d go no further \n till he d have satisfaction at Lanigan s Ball \n In the midst of the row Miss Kerrigan fainted \n her cheeks at the same time as red as a rose \n some of the boys decreed she was painted \n she took a wee drop too much I suppose \n Her sweetheart Ned Morgan all powerful and able \n when he saw his fair colleen stretched out by the wall \n he tore the left leg from under the table \n and smashed all the dishes at Lanigan s Ball'

sentences = data.lower().split('\n')

tokenizer.fit_on_texts(sentences)

num_words = len(tokenizer.word_index) + 1
num_sentences = len(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

x_sequences = []
y_labels = []
for token in sequences:
    x_sequences.append(token[:-1])
    y_labels.append(token[-1])

#e one-hot encoded the y_labels
y_labels = tf.keras.utils.to_categorical(y_labels, num_classes=num_words)

MAX_LEN = max([len(x_sequences[i]) for i in range(num_sentences)])
x_sequences = np.array(pad_sequences(x_sequences,padding= 'pre' ,maxlen=MAX_LEN))


#MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, 32, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(num_words, activation= 'softmax' )
])

model.compile(loss= 'categorical_crossentropy' ,optimizer= 'adam' ,metrics=[ 'accuracy' ])
model.summary()

NUM_EPOCHS = 500
hist = model.fit(x_sequences, y_labels, epochs=NUM_EPOCHS)


#Prediction

test_sentence =  'Marta likes Dublin but'
next_words_pred = 10

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
