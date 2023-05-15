from GRU_model import GRU_Config, GRU_Model
import numpy as np
import tensorflow as tf
import _pickle as cPickle
import os
#
config = GRU_Config()
dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
             dtype=np.float32)
model = GRU_Model(config, dummy_embeddings, mode="demo")

# # create the saver:
saver = tf.compat.v1.train.Saver()
data_dir = '/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/data/numpy_params'
#
with tf.compat.v1.Session() as sess:
#     # restore all model variables:
      params_dir = "coco/data/img_features_attention/transform_params"
      saver.restore(sess, "/Users/chandramaniyadav/Downloads/Image Captioning Project/coco2017/data/best_model")
#
#     # get the restored W_img and b_img:
      with tf.compat.v1.variable_scope("img_transform", reuse=True):
          W_img = tf.compat.v1.get_variable("W_img")
          b_img = tf.compat.v1.get_variable("b_img")
#
          W_img = sess.run(W_img)
          b_img = sess.run(b_img)
#
          transform_params = {}
          transform_params["W_img"] = W_img
          transform_params["b_img"] = b_img
          cPickle.dump(transform_params, open(os.path.join(data_dir, 
                                                                      "transform_params"),"wb"))