import numpy as np

def get_pre_trained_weights():
    embeddings = {}
    with open("glove.6B.100d.txt", 'r') as txtfile:
        lines = txtfile.readlines()
        for line in lines:
            line = line.split()
            word = line[0]
            embeddings[word] = np.array(line[1:])

    return embeddings
