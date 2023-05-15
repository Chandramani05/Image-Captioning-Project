#Code by Chandr@mani

#Splitting the images by their IDs so I can sort it later

import _pickle as cPickle
import os
import numpy as np

val_img_dir = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/val2017/'

# create a list of the paths to all val images:
val_img_paths = [val_img_dir + file_name for file_name in\
                 os.listdir(val_img_dir) if ".jpg" in file_name]


# define where all test images are located:
test_img_dir = "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/test2017/"
# create a list of the paths to all test images:
test_img_paths = [test_img_dir + file_name for file_name in\
                  os.listdir(test_img_dir) if ".jpg" in file_name]

# get all val img ids:
val_img_ids = np.array([])
for val_img_path in val_img_paths:
    img_name = val_img_path.split("/")[7]
    # Remove the ".jpg" extension from the image name
    image_id_str = img_name.split(".")[0]
    # Extract the numeric part of the image ID string
    image_id_numeric = int(image_id_str[3:])
    img_id = int(image_id_numeric)
    val_img_ids = np.append(val_img_ids, img_id)

# get all test img ids:
test_img_ids = np.array([])
for test_img_path in test_img_paths:
    img_name = test_img_path.split("/")[7]
    # Remove the ".jpg" extension from the image name
    image_id_str = img_name.split(".")[0]
    # Extract the numeric part of the image ID string
    image_id_numeric = int(image_id_str[4:])
    img_id = int(image_id_numeric)
    test_img_ids = np.append(test_img_ids, img_id)
    


# save the val img ids to disk:
cPickle.dump(val_img_ids, open(os.path.join("coco2017", "val_img_ids"), "wb"))
# save the test img ids to disk:
cPickle.dump(test_img_ids, open(os.path.join("coco2017", "test_img_ids"), "wb"))