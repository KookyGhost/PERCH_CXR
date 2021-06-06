import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# import tensorflow_addons as tfa
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import PIL
from read_tfrecord import Extract_TFrecord
from model import Model
from configparser import ConfigParser
import cv2

def main():
      config_file = "./config.ini"
      cp = ConfigParser()
      cp.read(config_file)
      tfrecord_folder = cp["GRAD-CAM"].get("tfrecord_folder")
      # dataset_name = cp["GRAD-CAM"].get("dataset_name")
      tf_record_name = cp["GRAD-CAM"].get("tf_record_name")
      saved_weights_path  = cp["GRAD-CAM"].get("saved_weights_path")
      class_names = cp["TRAIN"].get("class_names").split(",")
      print(class_names)
      class_to_visualize = cp["GRAD-CAM"].get("class_to_visualize")
      print(class_to_visualize)
      image_size = cp["TRAIN"].getint("IMAGE_SIZE")
      tf_record_dir = os.path.join(tfrecord_folder, tf_record_name)
      reader = Extract_TFrecord(image_size=image_size, batch_size=1, sharding=False, grad_cam=True)
      data = reader.get_dataset(augment=False, tf_record_dir=tf_record_dir, sharding=False, shuffle=False)

      m = Model(image_size=image_size, n_classes=len(class_names), imagenet=False)
      model1 = m.gradcam_model1()
      model1.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
      model1.load_weights(saved_weights_path, by_name=True)
      category_index= class_names.index(class_to_visualize)
      model2 =m.gradcam_model2(category_index=category_index)
      model2.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
      model2.load_weights(saved_weights_path, by_name=True)

      index = 0
      for d in tqdm(iter(data)):
            index += 1
            with tf.GradientTape() as tape:
                  earlier_pred = model1(tf.expand_dims(d[0],axis=0))
                  loss = tf.keras.backend.sum(model2(earlier_pred))
            grads_taped = tf.math.l2_normalize(tape.gradient(loss, earlier_pred))
            output, grads_taped = earlier_pred[0,:,:,:], grads_taped[0,:,:,:]
            weights =np.mean(grads_taped, axis = (0, 1))
            cam = np.zeros(output.shape[0:2], dtype = np.float32)
            for i, w in enumerate(weights):
                  cam += w * output[:, :, i]
            cam = cv2.resize(cam.numpy(), (image_size, image_size))
            cam = np.maximum(cam, 0)
            heatmap = cam / (np.max(cam)+tf.keras.backend.epsilon())
            cam = cv2.applyColorMap(np.uint8(255-255*heatmap), cv2.COLORMAP_JET)
            original_image = reader._deprocess_numpy_input(d[0].numpy())
            gradcam = np.uint8(0.7*original_image + 0.3*cam)
            mpimg.imsave("grad-cam"+str(index)+".png", gradcam)


if __name__ == "__main__":
    main()