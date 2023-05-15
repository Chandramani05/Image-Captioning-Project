import numpy as np
import _pickle as cPickle
import os


captions_dir = 'coco2017/annotations'
data_dir = 'coco2017'

#loading the valcubulary from the disk
vocabulary = cPickle.load(open(os.path.join(data_dir, "vocubalary"),"rb"))
vocab_size = len(vocabulary)

print(vocab_size)


# Reading all the words and their corresponding pretrained word vec from file : 
pretrained_words = []
word_vectors = []

word_vec_dim = 300

with open(os.path.join(captions_dir,"glove.6B.300d.txt")) as file :
    for line in file :
        #removing the new line char at the ned
        line  = line.strip()

        #seperate the word from the word_vector
        line_elements = line.split(" ")
        word = line_elements[0]
        word_vector = line_elements[1:]

        pretrained_words.append(word)
        word_vectors.append(word_vector)

#create an embedding matrix where row i is the pretrained word vector correspondng to word i in the vocab

embeddings_matrix = np.zeros((vocab_size, word_vec_dim))   

for vocab_index, word in enumerate(vocabulary) :
    if vocab_index % 1000 == 0:
        print(vocab_index)

    if word not in ["<SOS>", "<UNK>","<EOS>"] :
        word_embedd_index = pretrained_words.index(word)
        word_vector = word_vectors[word_embedd_index]

        #converting into numpy array and then to float
        word_vector = np.array(word_vector)
        word_vector = word_vector.astype(float)
        #adding to the matrix : 
        embeddings_matrix[vocab_index,:] = word_vector


print(embeddings_matrix)

#saving the embedding matrix to disk
cPickle.dump(embeddings_matrix, open(os.path.join(data_dir,"embeddings_matrix"), "wb"))


