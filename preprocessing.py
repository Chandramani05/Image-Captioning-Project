#Code By Chandr@mani
#Preprocessing All the Captions


import numpy as np
import _pickle as cPickle
import os
import re
import sys
sys.path.append(str('/Users/chandramaniyadav/Downloads/Image Captioning Project/coco'))
sys.path.append(str('/Users/chandramaniyadav/Downloads/Image Captioning Project/coco/coco-caption'))
sys.path.append(str('/Users/chandramaniyadav/Downloads/Image Captioning Project/coco/coco-caption/pycocoevalcap'))
print(sys.path)
from utilities import log

from pycocotools import COCO

captions_dir = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations'
data_dir = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017'

test_image_ids = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/test_img_ids'
val_image_ids = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/val_img_ids'

def getCaptions(type_of_data) : 
    captions_file = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations/captions_%s2017.json" % type_of_data
     
    #initialize COCO API for captions
    coco = COCO(captions_file)


    img_ids = coco.getImgIds()

    for step, img_id in enumerate(img_ids) :
        if step %1000 == 0 :
            print(step)

        caption_ids = coco.getAnnIds(img_id)
        caption_objs = coco.loadAnns(caption_ids)  

        #get Caption
        for caption_obj in caption_objs : 

            caption_id = caption_obj['id']
            caption_id_2_img_id[caption_id] = img_id
            caption = caption_obj['caption']

            caption = caption.strip()

            caption = caption.lower()

            caption = re.sub("[^a-z0-9]+", " ",caption)

            caption = re.sub("   ", " ", caption)
            # converting the caption to vector of words
            caption = caption.split(" ")
            # remove any empty chars still left in the caption:
            while " " in caption :
                index = caption.index(" ")
                del caption[index] 
            if str(img_id) in test_image_ids:
                test_caption_id_2_caption[caption_id] = caption
            elif str(img_id) in val_image_ids:
                val_caption_id_2_caption[caption_id] = caption
            else:
                train_caption_id_2_caption[caption_id] = caption       



train_caption_id_2_caption = {}
test_caption_id_2_caption = {}
val_caption_id_2_caption = {}
caption_id_2_img_id = {}
getCaptions("train")
getCaptions("val")


# save the caption_id to img_id mapping dict to disk:
cPickle.dump(caption_id_2_img_id,
        open(os.path.join(data_dir, "caption_id_2_img_id"), "wb"))



#Getting all the words that have a pretrained word embedding

pretrained_words = []

with open(os.path.join(captions_dir, "glove.6B.300d.txt")) as file :
    for line in file :
        line_elements = line.split(" ")
        word = line_elements[0]
        pretrained_words.append(word)


## Counting how many times each word occurs in the training set :
word_counts = {}

for caption_id in train_caption_id_2_caption : 
    caption = train_caption_id_2_caption[caption_id]

    for word in caption :
        if word not in word_counts :
            word_counts[word] = 1
        else :
            word_counts[word] += 1



vocubalary = []

for word in word_counts :
    word_count = word_counts[word]

    if word_count >=5 and word in pretrained_words :
        vocubalary.append(word)


#replace all the other word in with tags <UNK> <SOS > <EOS>
for step, caption_id in enumerate (train_caption_id_2_caption) :
    if step % 1000 == 0:
        print (step)

    caption = train_caption_id_2_caption[caption_id]
    for word_index in range(len(caption)) :
        word = caption[word_index]  
        if word not in vocubalary :
            caption[word_index] = "<UNK>"

    caption.insert(0, "<SOS>")
    caption.append("<EOS>")    

vocubalary.insert(0, "<UNK>")
vocubalary.insert(0,"<SOS>") 
vocubalary.insert(0,"<EOS>")   


#Dumpling the vocab to the folder

cPickle.dump(vocubalary, open(os.path.join(data_dir, "vocubalary"), "wb"))

for step, caption_id in enumerate (val_caption_id_2_caption) :
    if step % 1000 == 0:
        print (step)

    caption = val_caption_id_2_caption[caption_id]
    for word_index in range(len(caption)) :
        word = caption[word_index]  
        if word not in vocubalary :
            caption[word_index] = "<UNK>"

    caption.insert(0, "<SOS>")
    caption.append("<EOS>")


# caption with an <EOS> token:
for step, caption_id in enumerate(test_caption_id_2_caption):
    if step % 1000 == 0:
        print(step)


    caption = test_caption_id_2_caption[caption_id]
    # prepend the caption with an <SOS> token;
    caption.insert(0, "<SOS>")
    # append tge caption with an <EOS> token:
    caption.append("<EOS>")     



    #tokenized_caption = np.array(tokenized_caption)
    # save:
    #train_caption_id_2_caption[caption_id] = tokenized_caption


for step, caption_id in enumerate(train_caption_id_2_caption) :
    if step % 1000 == 0 :
        print (step)

    caption = train_caption_id_2_caption[caption_id]

    #tokenize the caption : 
    tokenized_caption = []
    for word in caption :
        word_index = vocubalary.index(word)
        tokenized_caption.append(word_index)
    
    #converting that into numpy array
    tokenized_caption  = np.array(tokenized_caption)

    train_caption_id_2_caption[caption_id] = tokenized_caption

 #save all the captions to disk :

cPickle.dump(train_caption_id_2_caption, open(os.path.join(data_dir, "train_caption_id_2_caption"),"wb"))
cPickle.dump(test_caption_id_2_caption, open(os.path.join(data_dir, "test_caption_id_2_caption"),"wb"))
cPickle.dump(val_caption_id_2_caption, open(os.path.join(data_dir, "val_caption_id_2_caption"), "wb"))


# map all train captions to their length:
train_caption_length_2_caption_ids = {}
for caption_id in train_caption_id_2_caption:
    caption = train_caption_id_2_caption[caption_id]
    caption_length = len(caption)
    if caption_length not in train_caption_length_2_caption_ids:
        train_caption_length_2_caption_ids[caption_length] = [caption_id]
    else:
        train_caption_length_2_caption_ids[caption_length].append(caption_id)

# map each train caption length to the number of captions of that length:
train_caption_length_2_no_of_captions = {}
for caption_length in train_caption_length_2_caption_ids:
    caption_ids = train_caption_length_2_caption_ids[caption_length]
    no_of_captions = len(caption_ids)
    train_caption_length_2_no_of_captions[caption_length] = no_of_captions


# Saving all the train data to disk

cPickle.dump(train_caption_length_2_no_of_captions, open(os.path.join(data_dir, 
                                                                      "train_caption_length_2_no_of_captions"),"wb"))
cPickle.dump(train_caption_length_2_caption_ids, open(os.path.join(data_dir, 
                                                                      "train_caption_length_2_caption_ids"),"wb"))

print(train_caption_length_2_no_of_captions)

# map all train captions to their length:
val_caption_length_2_caption_ids = {}
for caption_id in val_caption_id_2_caption:
    caption = val_caption_id_2_caption[caption_id]
    caption_length = len(caption)
    if caption_length not in val_caption_length_2_caption_ids:
        val_caption_length_2_caption_ids[caption_length] = [caption_id]
    else:
        val_caption_length_2_caption_ids[caption_length].append(caption_id)

# map each train caption length to the number of captions of that length:
val_caption_length_2_no_of_captions = {}
for caption_length in val_caption_length_2_caption_ids:
    caption_ids = val_caption_length_2_caption_ids[caption_length]
    no_of_captions = len(caption_ids)
    val_caption_length_2_no_of_captions[caption_length] = no_of_captions


# Saving all the train data to disk

cPickle.dump(val_caption_length_2_no_of_captions, open(os.path.join(data_dir, 
                                                                      "val_caption_length_2_no_of_captions"),"wb"))
cPickle.dump(val_caption_length_2_caption_ids, open(os.path.join(data_dir, 
                                                                      "val_caption_length_2_caption_ids"),"wb"))

print(val_caption_length_2_caption_ids)


