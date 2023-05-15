
import _pickle as cPickle
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import skimage.io as io
from datetime import datetime # to get the current date and time
import sys
sys.path.append(str('/Users/chandramaniyadav/Downloads/Image Captioning Project/coco/PythonAPI'))
sys.path.append(str('/Users/chandramaniyadav/Downloads/Image Captioning Project/coco/coco-caption'))

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
from pycocotools.coco import COCO

# add the "coco-caption" dir to the path so that "pycocoevalcap" can be found:
from pycocoevalcap.eval import COCOEvalCap

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def get_batches(model_obj):
    """
    - DOES: randomly shuffles all train caption ids and groups them into batches
    (of size model_obj.config.batch_size) where all captions in any given batch
    has the same length.
    """

    # get the batch size:
    batch_size = model_obj.config.batch_size

    # group all caption ids in batches:
    batches_of_caption_ids = []
    for caption_length in model_obj.caption_length_2_no_of_captions:
        caption_ids = model_obj.caption_length_2_caption_ids[caption_length]
        # randomly shuffle the order of the caption ids:
        random.shuffle(caption_ids)

        no_of_captions = model_obj.caption_length_2_no_of_captions[caption_length]
        no_of_full_batches = int(no_of_captions/batch_size)

        # add all full batches to batches_of_caption_ids:
        for i in range(no_of_full_batches):
            batch_caption_ids = caption_ids[i*batch_size:(i+1)*batch_size]
            batches_of_caption_ids.append(batch_caption_ids)

        # get the remaining caption ids and add to batches_of_captions (not a
        # full batch, i.e, it will contain fewer than "batch_size" captions):
        #batch_caption_ids = caption_ids[no_of_full_batches*batch_size:]
        #batches_of_caption_ids.append(batch_caption_ids)

    # randomly shuffle the order of the batches:
    random.shuffle(batches_of_caption_ids)

    return batches_of_caption_ids

def get_batch_ph_data(model_obj, batch_caption_ids):
    """
    - DOES: takes in a batch of caption ids, gets all corresponding data
    (img feature vectors and captions) and returns it in a format ready to be
    fed to the model (LSTM/GRU) placeholders in a feed_dict.
    """

    # get the dimension parameters:
    batch_size = model_obj.config.batch_size
    img_dim = model_obj.config.img_dim
    caption_length = len(model_obj.train_caption_id_2_caption[batch_caption_ids[0]])

    # initialize the return data:
    captions = np.zeros((batch_size, caption_length))
    # (row i of captions will be the tokenized caption for ex i in the batch)
    img_vectors = np.zeros((batch_size, img_dim))
    # (row i of img_vectors will be the img feature vector for ex i in the batch)
    labels = -np.ones((batch_size, caption_length + 1))
    # (row i of labels will be the targets for ex i in the batch)

    # populate the return data:
    for i in range(len(batch_caption_ids)):
        caption_id = batch_caption_ids[i]
        # get the img id:
        img_id = model_obj.caption_id_2_img_id[caption_id]
        # get the img's feature vector:
        if img_id in model_obj.train_img_id_2_feature_vector:
            img_vector = model_obj.train_img_id_2_feature_vector[img_id]
        else:
            img_vector = np.zeros((1, img_dim))
        # get the caption:
        caption = model_obj.train_caption_id_2_caption[caption_id]

        captions[i] = caption
        img_vectors[i] = img_vector
        labels[i, 1:caption_length] = caption[1:]

        # example to explain labels:
        # caption == [<SOS>, a, cat, <EOS>]
        # caption_length == 4
        # labels[i] == [-1, -1, -1, -1, -1]
        # caption[1:] == [a, cat, <EOS>]
        # labels[i, 1:caption_length] = caption[1:] gives:
        # labels[i] == [-1, a, cat, <EOS>, -1]
        # corresponds to the input:
        # img, <SOS>, a, cat, <EOS>
        # img: no prediciton should be made (-1)
        # <SOS>: should predict a (a)
        # a: should predict cat (cat)
        # cat: should predict <EOS> (<EOS>)
        # <EOS>: no prediction should be made (-1)

    return captions, img_vectors, labels

def get_batch_ph_data_attention(model_obj, batch_caption_ids):
    """
    - DOES: takes in a batch of caption ids, gets all corresponding data
    (img feature arrays and captions) and returns it in a format ready to be
    fed to the model (LSTM_attention/GRU_attention) placeholders in a feed_dict.
    """

    # get the dimension parameters:
    batch_size = model_obj.config.batch_size
    img_feature_dim = model_obj.config.img_feature_dim
    no_of_img_feature_vecs = model_obj.config.no_of_img_feature_vecs
    max_caption_length = model_obj.config.max_caption_length
    caption_length = len(model_obj.train_caption_id_2_caption[batch_caption_ids[0]])

    # initialize the return data:
    captions = np.zeros((batch_size, max_caption_length))
    # (row i of captions will be the tokenized (padded) caption for ex i in the batch)
    img_features = np.zeros((batch_size, no_of_img_feature_vecs, img_feature_dim))
    # (img_features[i] will be the (64x300) img feature array for ex i in the batch)
    labels = -np.ones((batch_size, max_caption_length))
    # (row i of labels will be the (padded) targets for ex i in the batch)

    # populate the return data:
    data_dir = 'coco2017'
    for i in range(len(batch_caption_ids)):
        caption_id = batch_caption_ids[i]
        # get the img id:
        img_id = model_obj.caption_id_2_img_id[caption_id]
        # get the img's feature array:
        try:
            img_feature_vectors = model_obj.img_id_2_feature_array[img_id]
        except:
            print ("zero-vector!")
            img_feature_vectors = np.zeros((no_of_img_feature_vecs,
                        img_feature_dim))
        # get the caption:
        caption = model_obj.train_caption_id_2_caption[caption_id]

        captions[i, 0:caption_length] = caption
        img_features[i] = img_feature_vectors
        labels[i, 0:caption_length-1] = caption[1:]

        # example to explain labels:
        # caption == [<SOS>, a, cat, <EOS>]
        # caption_length == 4
        # labels[i] == [-1, -1, -1, -1] (assume max_caption_length == caption_length)
        # caption[1:] == [a, cat, <EOS>]
        # labels[i, 0:caption_length-1] = caption[1:] gives:
        # labels[i] == [a, cat, <EOS>, -1]
        # corresponds to the input:
        # <SOS>, a, cat, <EOS>
        # <SOS>: should predict a (a)
        # a: should predict cat (cat)
        # cat: should predict <EOS> (<EOS>)
        # <EOS>: no prediction should be made (-1)
        # (if max_caption_length > caption_length, then labels[i] will be padded
        # with -1:s, which are masked in the loss computation)

    return captions, img_features, labels

def train_data_iterator(model_obj):
    """
    - DOES: groups all train caption ids into batches and then yields formated
    data for each batch to enable iteration. Used by the LSTM/GRU model.
    """

    # create batches of the train caption ids:
    batches_of_caption_ids = get_batches(model_obj)

    for batch_of_caption_ids in batches_of_caption_ids:
        # get the batch's data in a format ready to be fed into the placeholders:
        captions, img_vectors, labels = get_batch_ph_data(model_obj,
                    batch_of_caption_ids)

        # yield the data to enable iteration (will be able to do:
        # for (captions, img_vector, labels) in train_data_iterator(model_obj):)
        yield (captions, img_vectors, labels)

def train_data_iterator_attention(model_obj):
    """
    - DOES: groups all train caption ids into batches and then yields formated
    data for each batch to enable iteration. Used by the
    LSTM_attention/GRU_attention model.
    """

    # get the batches of caption ids:
    batches_of_caption_ids = get_batches(model_obj)

    for batch_of_caption_ids in batches_of_caption_ids:
        # get the batch's data in a format ready to be fed into the placeholders:
        captions, img_arrays, labels = get_batch_ph_data_attention(model_obj,
                    batch_of_caption_ids)

        # yield the data to enable iteration (will be able to do:
        # for (captions, img_vectors, labels) in train_data_iterator(config):)
        yield (captions, img_arrays, labels)

def detokenize_caption(tokenized_caption, vocabulary):
    """
    - DOES: receives a tokenized caption with <SOS> and <EOS> tags and converts
    it into a string of readable text.
    """

    caption_vector = []
    for word_index in tokenized_caption:
        # get the corresponding word:
        word = vocabulary[word_index]
        caption_vector.append(word)

    # remove <SOS> and <EOS>:
    caption_vector.pop(0)
    caption_vector.pop()

    # turn the caption vector into a string:
    caption = " ".join(caption_vector)

    return caption

def evaluate_captions(captions_file):
    """
    - DOES: computes the evaluation metrics BLEU-1 - BLEU4, CIDEr,
    METEOR and ROUGE_L for all captions in captions_file (generated on val or
    test imgs).
    """
    # define where the ground truth captions for the val (and test) imgs are located:
    true_captions_file = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations/captions_val2017.json"

    coco = COCO(true_captions_file)
    cocoRes = coco.loadRes(captions_file)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # set the imgs to be evaluated to the ones we have generated captions for:
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    # evaluate the captions (compute metrics):
    cocoEval.evaluate()
    # get the dict containing all computed metrics and metric scores:
    results_dict = cocoEval.eval

    return results_dict

def plot_performance(model_dir):
    """
    - DOES: plots the evaluation metrics and loss vs epoch for the model in
    model_dir. Also prints the top 5 CIDEr scores and their corresponding epoch.
    """

    # load the saved performance data with 'latin1' encoding:
    metrics_per_epoch = cPickle.load(open("%s/eval_results/metrics_per_epoch" % model_dir, 'rb'), encoding='latin1')
    loss_per_epoch = cPickle.load(open("%s/losses/loss_per_epoch" % model_dir, 'rb'), encoding='latin1')

    # separate the data for the different metrics:
    CIDEr_per_epoch = []
    Bleu_4_per_epoch = []
    ROUGE_L_per_epoch = []
    for epoch_metrics in metrics_per_epoch:
        CIDEr_per_epoch.append(epoch_metrics["CIDEr"])
        Bleu_4_per_epoch.append(epoch_metrics["Bleu_4"])
        ROUGE_L_per_epoch.append(epoch_metrics["ROUGE_L"])

    # plot the loss vs epoch:
    plt.figure(1)
    plt.plot(loss_per_epoch, "k^")
    plt.plot(loss_per_epoch, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("loss per epoch")
    plt.savefig("%s/plots/loss_per_epoch.png" % model_dir)

    # plot CIDEr vs epoch:
    plt.figure(2)
    plt.plot(CIDEr_per_epoch, "k^")
    plt.plot(CIDEr_per_epoch, "k")
    plt.ylabel("CIDEr")
    plt.xlabel("epoch")
    plt.title("CIDEr per epoch")
    plt.savefig("%s/plots/CIDEr_per_epoch.png" % model_dir)

    # plot Bleu_4 vs epoch:
    plt.figure(3)
    plt.plot(Bleu_4_per_epoch, "k^")
    plt.plot(Bleu_4_per_epoch, "k")
    plt.ylabel("Bleu_4")
    plt.xlabel("epoch")
    plt.title("Bleu_4 per epoch")
    plt.savefig("%s/plots/Bleu_4_per_epoch.png" % model_dir)

    # plot ROUGE_L vs epoch:
    plt.figure(4)
    plt.plot(ROUGE_L_per_epoch, "k^")
    plt.plot(ROUGE_L_per_epoch, "k")
    plt.ylabel("ROUGE_L")
    plt.xlabel("epoch")
    plt.title("ROUGE_L per epoch")
    plt.savefig("%s/plots/ROUGE_L_per_epoch.png" % model_dir)



    # print the top 5 CIDEr scores and their epoch number (to find the best
    # set of weights from the training process:)
    for i in range(5):
        max = np.max(np.array(CIDEr_per_epoch))
        arg_max = np.argmax(np.array(CIDEr_per_epoch))
        CIDEr_per_epoch[arg_max] = -1
        print ("%d: epoch %d, CIDEr score: %f" % (i+1, arg_max, max))

    print ("***********")

    # print the top 5 BLEU-4 scores and their epoch number (to find the best
    # set of weights from the training process:)
    for i in range(5):
        max = np.max(np.array(Bleu_4_per_epoch))
        arg_max = np.argmax(np.array(Bleu_4_per_epoch))
        Bleu_4_per_epoch[arg_max] = -1
        print ("%d: epoch %d, BLEU-4 score: %f" % (i+1, arg_max, max))


def compare_captions(model_dir, epoch, img_number):
    """
    - DOES: displays the ground truth captions and the generated caption (and the
    img) of img img_number in the val set for the model in model_dir at epoch.
    Allows for comparison of the generated and ground truth captions, as well as
    of the generated captions at different epochs.
    """

    # define where the ground truth captions for the val imgs are located:
    true_captions_file = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations/captions_val2017.json"

    coco = COCO(true_captions_file)
    # load the file containing all generated captions:
    cocoRes = coco.loadRes("%s/generated_captions/captions_%d.json"\
                % (model_dir, epoch))

    # get the img ids of all imgs for which captions have been generated:
    img_ids = cocoRes.getImgIds()
    # choose one specific img that you wish to study:
    img_id = img_ids[img_number]

    # print all ground truth captions for the img:
    print( "ground truth captions:")
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    # print the generated caption for the img:
    print ("generated caption:")
    annIds = cocoRes.getAnnIds(imgIds=img_id)
    anns = cocoRes.loadAnns(annIds)
    coco.showAnns(anns)

    # display the img:
    img = coco.loadImgs(img_id)[0]
    I = io.imread("/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/val2017/%s" % img["file_name"])
    plt.imshow(I)
    plt.axis('off')
    plt.show()

def get_max_caption_length(batch_size):
    """
    - DOES: returns the maximum length any caption will have if the train data
    is grouped into batches of size batch_size (since all captions in a batch
    has the same size and longer captions are more rare, this maximum length
    will decrease as the batch size is increased).
    """
    file_path = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/train_caption_length_2_no_of_captions"

    # Load the binary file using cPickle.load() with 'rb' mode
    with open(file_path, 'rb') as file:
        caption_length_2_no_of_captions = cPickle.load(file)


    # among the caption lengths that at least batch_size captions have, get the max:
    max_caption_length = 0
    for caption_length in caption_length_2_no_of_captions:
        no_of_captions = caption_length_2_no_of_captions[caption_length]
        if no_of_captions >= batch_size and caption_length > max_caption_length:
            max_caption_length = caption_length
    print("Max Caption Length : " + str(max_caption_length))
    return max_caption_length

def log(log_message):
    """
    - DOES: adds a log message "log_message" and its time stamp to a log file.
    """

    # open the log file and make sure that it's closed properly at the end of the
    # block, even if an exception occurs:
    with open("log.txt", "a") as log_file:
        # write the current time stamp and log message to logfile:
        log_file.write(datetime.strftime(datetime.today(),
                    "%Y-%m-%d %H:%M:%S") + ": " + log_message)
        log_file.write("\n") # (so the next message is put on a new line)

def plot_comparison_curves(model_dirs, metric, params_dict):
    """
    - DOES: plots "metric" (e.g. CIDEr score) vs. epoch number for all models
      specified by their directory in "model_dirs", in the same graph. Is used
      to create a comparison plot for different values of e.g. the dropout
      keep probability.
    """
    param = params_dict["param"]
    param_values = params_dict["param_values"]

    for model_dir, param_value in zip(model_dirs, param_values):
        if metric == "loss":
            loss_per_epoch = cPickle.load(open("%s/losses/loss_per_epoch" % model_dir))
            plt.plot(loss_per_epoch, label="%s: %s" %(param, param_value))

        elif metric in ["CIDEr", "Bleu_4", "ROUGE_L", "METEOR"]:
            metrics_per_epoch = cPickle.load(open("%s/eval_results/metrics_per_epoch"\
                % model_dir))

            # get the data per epoch for the specific metric:
            metric_per_epoch = []
            for epoch_metrics in metrics_per_epoch:
                metric_per_epoch.append(epoch_metrics[metric])
            # plot the data:
            plt.plot(metric_per_epoch, label="%s: %s" %(param, param_value))

    plt.ylabel(metric, fontsize=15)
    plt.xlabel("epoch", fontsize=15)
    plt.title("", fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig("comparison_plot")

def evaluate_base_model():
    """
    - DOES: generates captions for all 5000 imgs in test using the most basic
      baselime model (nearest neighbor), evaluates the captions and returns the
      metric scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, METEOR and ROUGE_L).
    """

    print ("loading data")

    # get the map from test img id to feature vector:
    test_img_id_2_feature_vector =\
                cPickle.load(open("/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/test_img_id_2_feature_vector"), )
    # turn the map into a list of tuples (to make it iterable):
    test_set = test_img_id_2_feature_vector.items()

    # get the map from train img id to feature vector:
    train_img_id_2_feature_vector =\
                cPickle.load(open("/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/train_img_id_2_feature_vector"))
    # turn the map into a list of tuples (to make it iterable):
    train_set = train_img_id_2_feature_vector.items()

    # define where the ground truth captions for the train imgs are located:
    true_captions_file1 = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations/captions_val2017.json"
    true_captions_file2 = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations/captions_train2017.json"
    # initialize coco objects:
    coco1 = COCO(true_captions_file1)
    coco2 = COCO(true_captions_file2)

    print ("all data is loaded")

    # generate a caption for every test image:
    captions = []
    for step, (img_id, img_vector) in enumerate(test_set):
        if step % 100 == 0:
            print ("generating captions on test: %d" % step)

        distances = []
        ids = []
        for (train_img_id, train_img_vector) in train_set:
            distance = np.linalg.norm(img_vector - train_img_vector)
            distances.append(distance)
            ids.append(train_img_id)

        # get the img id of the train img that is closest to the img:
        closest_img_id = ids[np.argmin(distances)]

        # randomly pick one of the captions for the closest train img:
        annIds = coco1.getAnnIds(imgIds=closest_img_id)
        anns = coco1.loadAnns(annIds)
        if anns == []:
            annIds = coco2.getAnnIds(imgIds=closest_img_id)
            anns = coco2.loadAnns(annIds)
        random.shuffle(anns)
        img_caption = anns[0]["caption"]

        # save the generated caption together with the img id in the format
        # expected by the COCO evaluation script:
        caption_obj = {}
        caption_obj["image_id"] = img_id
        caption_obj["caption"] = img_caption
        captions.append(caption_obj)

    # save the captions as a json file (will be used by the eval script):
    captions_file = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/annotations/test_captions.json"
    with open(captions_file, "w") as file:
        json.dump(captions, file, sort_keys=True, indent=4)

    # evaluate the generated captions:
    results_dict = evaluate_captions(captions_file)

    print (results_dict)

def main():
    test = 1

if __name__ == '__main__':
    main()
