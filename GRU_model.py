"""
- ASSUMES: that preprocess_captions.py, extract_img_features.py and
  create_initial_embeddings.py has already been run.
- DOES: defines the GRU model and contains a script for training the model.
"""


import numpy as np
import tensorflow as tf

import _pickle as cPickle
import os 
import time
import json
import random

from utilities import train_data_iterator, detokenize_caption, evaluate_captions
from utilities import plot_performance



data_dir = 'coco2017'
class GRU_Config(object) :

    def __init__(self, debug = False):
        self.dropout = 0.75 # (keep probability)
        self.embed_dim = 300 # (dimension of word embeddings)
        self.hidden_dim = 400 # (dimension of hidden state)
        self.batch_size = 256
        self.lr = 0.001
        self.img_dim = 2048 # (dimension of img feature vectors)
        self.vocab_size = 10164 # (no of words in the vocabulary)
        self.no_of_layers = 1 # (no of layers in the RNN)
        if debug:
            self.max_no_of_epochs = 2
        else:
            self.max_no_of_epochs = 51
        self.max_caption_length = 40
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim, self.embed_dim,
                    self.no_of_layers)
        self.model_dir = "models/GRUs/%s" % self.model_name


class GRU_Model(object) :

    def __init__(self, config, GloVe_embeddings, debug = False, mode = 'training'):
        #inititalizing some parametrs and adding everything to the computational graph

        self.GloVe_embeddings = GloVe_embeddings
        self.debug = debug
        self.config = config

        if mode != "demo" :
            # create all dirs for saving weights and eval results:
            self.create_model_dirs()
            # load all data from disk needed for training:
            self.load_utilities_data()
        # add placeholders to the comp graph:
        self.add_placeholders()
        # transform the placeholders and add the final model input to the graph:
        self.add_input()
        # compute logits (unnormalized prediction probs) and add to the graph:
        self.add_logits()
        if mode != "demo":
            # compute the batch loss and add to the graph:
            self.add_loss_op()
            # add a training operation (for optimizing the loss) to the graph:
            self.add_training_op()


    def create_model_dirs(self):
        # create the main model directory:
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        # create the dir where model weights will be saved during training:
        if not os.path.exists(os.path.join(self.config.model_dir, "weights")):
            os.mkdir(os.path.join(self.config.model_dir, "weights"))

        # create the dir where generated captions will be saved during training:
        if not os.path.exists(os.path.join(self.config.model_dir, "generated_captions")):
            os.mkdir(os.path.join(self.config.model_dir, "generated_captions"))

        # create the dir where epoch losses will be saved during training:
        if not os.path.exists(os.path.join(self.config.model_dir, "losses")):
            os.mkdir(os.path.join(self.config.model_dir, "losses"))

        # create the dir where evaluation metrics will be saved during training:
        if not os.path.exists(os.path.join(self.config.model_dir, "eval_results")):
            os.mkdir(os.path.join(self.config.model_dir, "eval_results"))

        # create the dir where performance plots will be saved after training:
        if not os.path.exists(os.path.join(self.config.model_dir, "plots")):
            os.mkdir(os.path.join(self.config.model_dir, "plots"))


    def load_utilities_data(self):
        """
        - DOES: loads all data from disk (vocabulary, img feature vectors etc.)
        needed for training.
        """

        print ("loading utilities data...")
        

        # load the vocabulary:
        self.vocabulary = cPickle.load(open(os.path.join(data_dir, "vocubalary"),"rb"))

        # load data to map from caption id to img feature vector:
        self.caption_id_2_img_id =\
                    cPickle.load(open(os.path.join(data_dir, "caption_id_2_img_id"),"rb"))
        if self.debug:
            self.train_img_id_2_feature_vector =\
                    cPickle.load(open(os.path.join(data_dir, "val_img_id_2_feature_vector"),"rb"))
        else:
            self.train_img_id_2_feature_vector =\
                    cPickle.load(open(os.path.join(data_dir, "train_img_id_2_feature_vector"),"rb"))

        # load data to map from caption id to caption:
        if self.debug:
            self.train_caption_id_2_caption =\
                    cPickle.load(open(os.path.join(data_dir, "val_caption_id_2_caption"),"rb"))
        else:
            self.train_caption_id_2_caption =\
                    cPickle.load(open(os.path.join(data_dir, "train_caption_id_2_caption"),"rb"))

        # load data needed to create batches:
        if self.debug:
            self.caption_length_2_caption_ids =\
                cPickle.load(open(os.path.join(data_dir, "val_caption_length_2_caption_ids"),"rb"))
            self.caption_length_2_no_of_captions =\
                 cPickle.load(open(os.path.join(data_dir, "val_caption_length_2_no_of_captions"),"rb"))
        else:
            self.caption_length_2_caption_ids =\
                cPickle.load(open(os.path.join(data_dir, "train_caption_length_2_caption_ids"),"rb"))
            self.caption_length_2_no_of_captions =\
                cPickle.load(open(os.path.join(data_dir, "train_caption_length_2_no_of_captions"),"rb"))
        print ("all utilities data is loaded!")

    def add_placeholders(self):
        """
        - DOES: adds placeholders for captions, imgs, labels and keep_prob to
        the computational graph. These placeholders will be fed actual data
        corresponding to each batch during training.
        """
        tf.compat.v1.disable_eager_execution()
        # add the placeholder for the batch captions (row i of caption_ph will
        # be the tokenized caption for ex i in the batch):
        self.captions_ph = tf.compat.v1.placeholder(tf.int32,
                    shape=[None, None], # ([batch_size, caption_length])
                    name="captions_ph")
        # add the placeholder for the batch imgs (row i of imgs_ph will be the
        # img feature vector for ex i in the batch):
        self.imgs_ph = tf.compat.v1.placeholder(tf.float32,
                    shape=[None, self.config.img_dim], # ([batch_size, img_dim])
                    name="imgs_ph")
        # add the placeholder for the batch labels (row i of labels_ph will
        # be the labels/targets for ex i in the batch):
        self.labels_ph = tf.compat.v1.placeholder(tf.int32,
                    shape=[None, None], # ([batch_size, caption_length+1])
                    name="labels_ph")
        # add the placeholder for the keep_prob (with what probability we will
        # keep a hidden unit during training):
        self.dropout_ph = tf.compat.v1.placeholder(tf.float32, name="dropout_ph") # (keep_prob)

    def create_feed_dict(self, captions_batch, imgs_batch, labels_batch=None, dropout=1):
        """
        - DOES: returns a feed_dict mapping the placeholders to the actual
        input data (this is how we run the network on specific data).
        """

        feed_dict = {}
        feed_dict[self.captions_ph] = captions_batch
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.dropout_ph] = dropout
        if labels_batch is not None:
            # only add the labels data if it's specified (during caption
            # generation, we won't have any labels):
            feed_dict[self.labels_ph] = labels_batch

        return feed_dict   

    def add_input(self):
        """
        - DOES: transforms the imgs_ph to a tensor of shape
        [batch_size, 1, embed_dim], gets the word vector for each tokenized word
        in captions_ph giving a tensor of shape
        [batch_size, caption_length, embed_dim], and finally concatenates the
        two into a tensor of shape [batch_size, caption_length+1, embed_dim].
        This tensor is the input to the network, meaning that we will feed in
        the img, then <SOS>, then each word in the caption, and then
        finally <EOS>.
        """

        # transform img_ph into a tensor of shape [batch_size, 1, embed_dim]:
        with tf.compat.v1.variable_scope("img_transform"):
            # initialize the transform parameters:
            W_img = tf.compat.v1.get_variable("W_img",
                        shape=[self.config.img_dim, self.config.embed_dim],
                        initializer=tf.initializers.GlorotUniform())
            b_img = tf.compat.v1.get_variable("b_img", shape=[1, self.config.embed_dim],
                        initializer=tf.constant_initializer(0))
            # tranform img_ph to shape [batch_size, embed_dim]:
            imgs_input = tf.nn.sigmoid(tf.matmul(self.imgs_ph, W_img) + b_img)
            # reshape into shape [batch_size, 1, embed_dim]:
            imgs_input = tf.expand_dims(imgs_input, 1)

        # get the word vector for each tokenized word in captions_ph:
        with tf.compat.v1.variable_scope("captions_embed"):
            # initialize the embeddings matrix with pretrained GloVe vectors (
            # note that we will train the embeddings matrix as well!):
            word_embeddings = tf.compat.v1.get_variable("word_embeddings",
                        initializer=self.GloVe_embeddings)
            # get the word vectors (gives a tensor of shape
            # [batch_size, caption_length, embed_dim]):
            captions_input = tf.nn.embedding_lookup(word_embeddings,
                        self.captions_ph)

        # concatenate imgs_input and captions_input to get the final input (has
        # shape [batch_size, caption_length+1, embed_dim])
        self.input = tf.concat([imgs_input, captions_input], axis=1)



    def add_logits(self):
        """
        - DOES: feeds self.input through a GRU, producing a hidden state vector
        for each word/img and computes all corresponding logits (unnormalized
        prediction probabilties over the vocabulary, a softmax step but without
        the actual softmax).
        """

        # create a GRU cell:
        GRU_cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.config.hidden_dim)
        # apply dropout to the GRU cell:
        GRU_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(GRU_cell,
                    input_keep_prob=self.dropout_ph,
                    output_keep_prob=self.dropout_ph)
        # stack no_of_layers GRU cells on top of each other (for a deep GRU):
        stacked_GRU_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [GRU_cell]*self.config.no_of_layers)
        # initialize the state of the stacked GRU cell (tf.shape(self.input)[0]
        # gets the current batch size) (the state contains both h and c for all
        # layers, thus its format is not trivial):
        initial_state = stacked_GRU_cell.zero_state(tf.shape(self.input)[0],
                    tf.float32)

        # feed self.input trough the stacked GRU cell and get the (top) hidden
        # state vector for each word/img returned in outputs (which has shape
        # [batch_size, caption_length+1, hidden_dim]) (final_state contains
        # h and c for all layers at the final timestep, not relevant here):
        outputs, final_state = tf.compat.v1.nn.dynamic_rnn(stacked_GRU_cell,
                    self.input, initial_state=initial_state)
        # reshape outputs into shape [batch_size*(caption_length+1), hidden_dim]
        # (outputs[0]: h for the img in ex 1 in the batch, outputs[1]: h for <SOS>
        # in ex 1 in the batch etc.):
        outputs = tf.reshape(outputs, [-1, self.config.hidden_dim])

        # compute corresponding logits for each hidden state vector in outputs,
        # resulting in a tensor self.logits of shape
        # [batch_size*(caption_length+1), vocab_size] (each word in self.input
        # will have a corr. logits vector, which is an unnorm. prob. distr. over
        # the vocab. The largets element corresponds to the predicted next word):
        with tf.compat.v1.variable_scope("logits"):
            # initialize the transform parametrs:
            W_logits = tf.compat.v1.get_variable("W_logits",
                        shape=[self.config.hidden_dim, self.config.vocab_size],
                        initializer=tf.initializers.GlorotUniform())
            b_logits = tf.compat.v1.get_variable("b_logits",
                        shape=[1, self.config.vocab_size],
                        initializer=tf.constant_initializer(0))
            # compute the logits:
            self.logits = tf.matmul(outputs, W_logits) + b_logits

    def add_loss_op(self):
        """
        - DOES: computes the CE loss for the batch.
        """

                # reshape labels_ph into shape [batch_size*(caption_length+1), ] (to
        # match the shape of self.logits):
        labels = tf.reshape(self.labels_ph, [-1])

        # remove all -1 labels and their corresponding logits (-1 labels
        # correspond to the img or <EOS> step, the predicitons at these
        # steps are irrelevant and should not contribute to the loss):
        mask = tf.greater_equal(labels, 0)
        masked_labels = tf.boolean_mask(labels, mask)
        masked_logits = tf.boolean_mask(self.logits, mask)

        # compute the CE loss for each word in the batch:
        loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=masked_labels, logits=masked_logits)
        # average the loss over all words to get the batch loss:
        loss = tf.reduce_mean(loss_per_word)

        self.loss = loss


    def add_training_op(self):
        """
        - DOES: creates a training operator for optimizing the loss.
        """

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def run_epoch(self, session):
        """
        - DOES: runs one epoch, i.e., for each batch it: computes the batch loss
        (forwardprop), computes all gradients w.r.t to the batch loss and updates
        all network variables/parameters accordingly (backprop).
        """

        batch_losses = []
        for step, (captions, imgs, labels) in enumerate(train_data_iterator(self)):
            # create a feed_dict with the batch data:
            feed_dict = self.create_feed_dict(captions, imgs,
                        labels_batch=labels, dropout=self.config.dropout)
            # compute the batch loss and compute & apply all gradients w.r.t to
            # the batch loss (without self.train_op in the call, the network
            # would not train, we would only compute the batch loss):
            batch_loss, _ = session.run([self.loss, self.train_op],
                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)

            if step % 100 == 0:
                print ("batch: %d | loss: %f" % (step, batch_loss))

            if step > 5 and self.debug:
                break

        # return a list containing the batch loss for each batch:
        return batch_losses
    

    def generate_img_caption(self, session, img_vector, vocabulary):
        """
        - DOES: generates a caption for the img feature vector img_vector.
        """

        # initialize the caption as "<SOS>":
        caption = np.zeros((1, 1))
        caption[0] = np.array(vocabulary.index("<SOS>"))
        # format img_vector so it can be fed to the network:
        img = np.zeros((1, self.config.img_dim))
        img[0] = img_vector

        # we will get one vector of logits for each timestep, element 0 corr. to
        # the img, element 1 corr. to <SOS> etc., to begin we want to get the
        # one corr. to "<SOS>":
        prediction_index = 1

        # predict the next word given the img and the current caption until we
        # get "<EOS>" or the caption length hits a max value:
        while int(caption[0][-1]) is not vocabulary.index("<EOS>") and\
                    caption.shape[1] < self.config.max_caption_length:
            feed_dict = self.create_feed_dict(caption, img)
            logits = session.run(self.logits, feed_dict=feed_dict)
            # (logits[0] = logits vector corr. to the img, logits[1] = logits
            # vector corr. to <SOS> etc.)

            # get the logits vector corr. to the last word in the current caption
            # (it gives what next word we will predict):
            prediction_logits = logits[prediction_index]
            # get the index of the predicted word (the word in the vocabulary
            # with the largest (unnormalized) probability):
            predicted_word_index = np.argmax(prediction_logits)
            # add the new word to the caption:
            new_word_col = np.zeros((1, 1))
            new_word_col[0] = np.array(predicted_word_index)
            caption = np.append(caption, new_word_col, axis=1)
            # increment prediction_index so that we'll look at the new last word
            # of the caption in the next iteration:
            prediction_index += 1

        # get the generated caption and convert to ints:
        caption = caption[0].astype(int)
        # convert the caption to actual text:
        caption = detokenize_caption(caption, vocabulary)

        return caption
    

    def generate_captions_on_val(self, session, epoch, vocabulary, val_set_size=5000):
        """
        - DOES: generates a caption for each of the first val_set_size imgs in
        the val set, saves them in the format expected by the provided COCO
        evaluation script and returns the name of the saved file.
        """

        if self.debug:
            val_set_size = 101

        # get the map from img id to feature vector:
        val_img_id_2_feature_vector =\
                    cPickle.load(open(os.path.join(data_dir, "val_img_id_2_feature_vector"),"rb"))
        # turn the map into a list of tuples (to make it iterable):
        val_img_id_feature_vector_list = val_img_id_2_feature_vector.items()
        # take the first val_set_size val imgs:
        val_set = list(val_img_id_feature_vector_list)[0:val_set_size]

        captions = []
        for step, (img_id, img_vector) in enumerate(val_set):
            if step % 100 == 0:
                print ("generating captions on val: %d" % step)

            # generate a caption for the img:
            img_caption = self.generate_img_caption(session, img_vector, vocabulary)

            print(img_caption, " ID : ", str(img_id))
            # save the generated caption together with the img id in the format
            # expected by the COCO evaluation script:
            caption_obj = {}
            caption_obj["image_id"] = img_id
            caption_obj["caption"] = img_caption
            captions.append(caption_obj)

        # save the captions as a json file (will be used by the eval script):
        captions_file = "%s/generated_captions/captions_%d.json"\
                    % (self.config.model_dir, epoch)
        with open(captions_file, "w") as file:
            json.dump(captions, file, sort_keys=True, indent=4)

        # return the name of the json file:
        return captions_file
    
def main():
    # create a config object:
    config = GRU_Config()
    # get the pretrained embeddings matrix:
   # Specify the path of the binary file to load
    file_path = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/embeddings_matrix"

    # Load the binary file using cPickle.load() with 'rb' mode
    with open(file_path, 'rb') as file:
        GloVe_embeddings = cPickle.load(file)
    GloVe_embeddings = GloVe_embeddings.astype(np.float32)
    # create a GRU model object:
    model = GRU_Model(config, GloVe_embeddings)

    # initialize the list that will contain the loss for each epoch:
    loss_per_epoch = []
    # initialize the list that will contain all evaluation metrics (BLEU, CIDEr,
    # METEOR and ROUGE_L) for each epoch:
    eval_metrics_per_epoch = []

    # create a saver for saving all model variables/parameters:
    saver = tf.compat.v1.train.Saver(max_to_keep=model.config.max_no_of_epochs)

    with tf.compat.v1.Session() as sess:
        # initialize all variables/parameters:
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        last_epoch_loss = 10
        

        for epoch in range(config.max_no_of_epochs):
            print ("###########################")
            print ("######## NEW EPOCH ########")
            print ("###########################")
            print ("epoch: %d/%d" % (epoch, config.max_no_of_epochs-1))

            # run an epoch and get all batch losses:
            batch_losses = model.run_epoch(sess)

            # compute the epoch loss:
            epoch_loss = np.mean(batch_losses)
            # save the epoch loss:
            loss_per_epoch.append(epoch_loss)
            # save the epoch losses to disk:
            with open("%s/losses/loss_per_epoch" % model.config.model_dir, "wb") as f:
                cPickle.dump(loss_per_epoch, f)

            if epoch_loss < last_epoch_loss:
                saver.save(sess, '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/data/best_model')
                last_epoch_loss = epoch_loss  

            # generate captions on a (subset) of val:
            captions_file = model.generate_captions_on_val(sess, epoch,
                        model.vocabulary, val_set_size=1000)
            # evaluate the generated captions (compute eval metrics):
            eval_result_dict = evaluate_captions(captions_file)
            # save the epoch evaluation metrics:
            eval_metrics_per_epoch.append(eval_result_dict)
            # save the evaluation metrics for all epochs to disk:
            # Replace cPickle with pickle and use binary write mode "wb" instead of "w"
            cPickle.dump(eval_metrics_per_epoch, open("%s/eval_results/metrics_per_epoch" % model.config.model_dir, "wb"))

            if eval_result_dict["CIDEr"] > 0.70:
                # save the model weights to disk:
                saver.save(sess, "%s/weights/model" % model.config.model_dir,
                            global_step=epoch)

            print ("epoch loss: %f | BLEU4: %f  |  CIDEr: %f" % (epoch_loss,
                        eval_result_dict["Bleu_4"], eval_result_dict["CIDEr"]))
        
              

    # plot the loss and the different evaluation metrics vs epoch:
    plot_performance(config.model_dir)

if __name__ == '__main__':
        main()


        

