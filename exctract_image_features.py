import os
import re
import tensorflow.compat.v1 as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import _pickle as cPickle
tf.disable_v2_behavior()

def load_pretrained_CNN():

    model_dir = 'inception'
    path_to_saved_model = os.path.join(model_dir, '/Users/chandramaniyadav/Downloads/Image Captioning Project/inception/classify_image_graph_def.pb')

    with gfile.FastGFile(path_to_saved_model, "rb") as model_file :
        #create an empty graph object
        graph_def = tf.GraphDef()

        #import the model definations :
        graph_def.ParseFromString(model_file.read())
        _ = tf.import_graph_def(graph_def, name = "")


def extract_img_features (img_paths, demo = False) :

    img_id_2_feature_vector = {}
     
    #load the CNN
    load_pretrained_CNN()

    with tf.Session() as sess : 
        #get the second last layer of Inception V3 Model (this
        # is what will be use feature vector for each image) :

        second_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for step, img_path in enumerate (img_paths) :
            if step % 100 == 0 :
                print(step)

            #read the image
            img_data = gfile.FastGFile(img_path, "rb").read()

            try:
                #getting tehe feature vector
                feature_vector = sess.run(second_to_last_tensor, 
                                          feed_dict = {"DecodeJpeg/contents:0": img_data})
            except:
                print("JPEG Error")
                print (img_path) 
                print('**' * 100)

            else :
                feature_vector = np.squeeze(feature_vector)

                if not demo:
                    # get the image id:
                    img_name = img_path.split("/")[7]
                    # Remove the ".jpg" extension from the image name
                    image_id_str = img_name.split(".")[0]
                    # Extract the numeric part of the image ID string
                    num = 4
                    if 'val' in image_id_str :
                        num = 3 
                    elif 'train' in image_id_str :
                        num = 5    

                    image_id_numeric = int(image_id_str[num:])
                    img_id = int(image_id_numeric)
                    print(img_id)
                else: # (if demo:)
                    # we're only extracting features for one img, (arbitrarily)
                    # set the img id to 0:
                    img_id = 0

                # save the feature vector and the img id:
                img_id_2_feature_vector[img_id] = feature_vector

        return img_id_2_feature_vector
    

def main () :
    #define where images are located :
    val_img_dir = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/val2017/'

    val_img_paths = [val_img_dir + file_name for file_name in\
                     os.listdir(val_img_dir) if ".jpg" in file_name]
    
    # get the feature vectors for all val imgs:
    val_img_id_2_feature_vector = extract_img_features(val_img_paths) 

    print (val_img_id_2_feature_vector)
        # save on disk:
    save_path = os.path.join("coco2017", "val_img_id_2_feature_vector")

    # save the feature vectors on disk using cPickle:
    with open(save_path, "wb") as file:
        cPickle.dump(val_img_id_2_feature_vector, file)
        print ("val done!")

    # define where all test imgs are located:
    test_img_dir = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/test2017/"
    # create a list of the paths to all test imgs:
    test_img_paths = [test_img_dir + file_name for file_name in\
                      os.listdir(test_img_dir) if ".jpg" in file_name]

    # get the feature vectors for all test imgs:
    test_img_id_2_feature_vector = extract_img_features(test_img_paths)

    # specify the path to save the feature vectors:
    save_path = os.path.join("coco2017", "test_img_id_2_feature_vector")

    # save the feature vectors on disk using cPickle:
    with open(save_path, "wb") as file:
        cPickle.dump(test_img_id_2_feature_vector, file)
        print ("test done!")

    # define where all train imgs are located:
    train_img_dir = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/train2017/"
    # create a list of the paths to all train imgs:
    train_img_paths = [train_img_dir + file_name for file_name in\
                       os.listdir(train_img_dir) if ".jpg" in file_name]
    # get the feature vectors for all train imgs:
    train_img_id_2_feature_vector = extract_img_features(train_img_paths)
    # specify the path to save the feature vectors:
    save_path = os.path.join("coco2017", "train_img_id_2_feature_vector")

    # save the feature vectors on disk using cPickle:
    with open(save_path, "wb") as file:
        cPickle.dump(train_img_id_2_feature_vector, file)
        print ("train done!")



     
      

if __name__ == '__main__':
    main()      