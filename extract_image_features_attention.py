

"""
- ASSUMES: that the image dataset has been manually split such that all train
  images are stored in "coco/images/train/", all test images are stored in
  "coco/images/test/" and all val images are stored in "coco/images/val". That
  the Inception-V3 model has been downloaded and placed in inception. That the
  dict numpy_params (containing W_img and b_img taken from the img_transform
  step in a well-performing non-attention model) is placed in
  coco/data/img_features_attention/transform_params.
- DOES: extracts a 64x300 feature array (64 300 dimensional feature vectors,
  one each for 8x8 different img regions) for each train/val/test img and saves
  each individual feature array to disk (to coco/data/img_features_attention).
  Is used in the attention models.
"""

import os
import re
import tensorflow.compat.v1 as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import _pickle as cPickle
from utilities import log

from exctract_image_features import load_pretrained_CNN

def extract_img_features_attention(img_paths, demo=False):
    """
    - Runs every image in "img_paths" through the pretrained CNN and
    saves their respective feature array (the third-to-last layer
    of the CNN transformed to 64x300) to disk.
    """

    # load the Inception-V3 model:
    load_pretrained_CNN()
    img_id_2_feature_vector = {}

    # load the parameters for the feature vector transform:
    file_path = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/data/numpy_params/transform_params"

# Load the data from the binary file
    with open(file_path, "rb") as f:
        transform_params = cPickle.load(f)  # Use 'pickle.load()' in Python 3

    print("Data loaded successfully from: {}".format(file_path))
    W_img = transform_params["W_img"]
    b_img = transform_params["b_img"]

    with tf.Session() as sess:
        # get the third-to-last layer in the Inception-V3 model (a tensor
        # of shape (1, 8, 8, 2048)):
        img_features_tensor = sess.graph.get_tensor_by_name("mixed_10/join:0")
        # reshape the tensor to shape (64, 2048):
        img_features_tensor = tf.reshape(img_features_tensor, (64, 2048))

        # apply the img transorm (get a tensor of shape (64, 300)):
        linear_transform = tf.matmul(img_features_tensor, W_img) + b_img
        img_features_tensor = tf.nn.sigmoid(linear_transform)

        for step, img_path in enumerate(img_paths):
            if step % 1000 == 0:
                print (step)

            # read the image:
            img_data = gfile.FastGFile(img_path, "rb").read()
            try:
                # get the img features (np array of shape (64, 300)):
                img_features = sess.run(img_features_tensor,
                        feed_dict={"DecodeJpeg/contents:0": img_data})
                #img_features = np.float16(img_features)
            except:
                print ("JPEG error for:")
                print (img_path)
            else:
                img_features = np.squeeze(img_features)
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
                else: # (if demo:)
                    # we're only extracting features for one img, (arbitrarily)
                    # set the img id to 0:
                    img_id = 0
         
                # save the img features to disk:

                # Specify the path to save the dumped data
                save_path = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/img_feature_attention"

                # Create the directory if it doesn't exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Dump the img_features data using cPickle.dump() function
                #with open(os.path.join(save_path, "%d" % img_id), "wb") as f:
                 #   cPickle.dump(img_features, f)  # Use 'pickle.dump()' in Python 3
                img_id_2_feature_vector[img_id] = img_features 

        return img_id_2_feature_vector            

def main():
    # define where all val imgs are located:
    val_img_dir = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/val2017/"
    # create a list of the paths to all val imgs:
    val_img_paths = [val_img_dir + file_name for file_name in\
                     os.listdir(val_img_dir) if ".jpg" in file_name]

    # define where all test imgs are located:
    test_img_dir = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/test2017/"
    # create a list of the paths to all test imgs:
    test_img_paths = [test_img_dir + file_name for file_name in\
                      os.listdir(test_img_dir) if ".jpg" in file_name]

    # define where all train imgs are located:
    train_img_dir = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/train2017/"
    # create a list of the paths to all train imgs:
    train_img_paths = [train_img_dir + file_name for file_name in\
                       os.listdir(train_img_dir) if ".jpg" in file_name]

    # create a list of the paths to all imgs:
    img_paths = val_img_paths + test_img_paths + train_img_paths

    # extract all features:
    img_id_2_feature_vector =extract_img_features_attention(img_paths)
    save_path = os.path.join("coco2017", "img_id_2_feature_vector")
    with open(save_path, "wb") as file:
        cPickle.dump(img_id_2_feature_vector, file)
        print ("done!")

if __name__ == '__main__':
    main()