import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from read_tfrecord import Extract_TFrecord
from configparser import ConfigParser
from model import Model
from callback import MultipleClassAUROC
from class_weights import get_class_weights, get_sample_counts
from loss import CustomLoss


config_file = "./config.ini"
cp = ConfigParser()
cp.read(config_file)
dataset_test_name = cp["DATA"].get("dataset_test_csv_file")
sharding = cp["DATA"].get("sharding")
output_dir = cp["DATA"].get("output_dir")
image_size = cp["TRAIN"].getint("IMAGE_SIZE")
class_names = cp["TRAIN"].get("class_names").split(",")
saved_weights_path  = cp["TEST"].get("saved_weights_path")
batch_size = cp["TEST"].getint("batch_size")
dataset_name = cp["DATA"].get("dataset_name")



def main():
      reader = Extract_TFrecord(image_size=image_size, batch_size=batch_size, sharding=sharding)
      tf_record_dir_test = os.path.join(output_dir, dataset_name+'_test')
      test = reader.get_dataset(augment=False, tf_record_dir=tf_record_dir_test, sharding=False, shuffle=False)
      y_true = reader.get_test_label(reader.read_tfrecord(tf_record_dir_test, sharding=False, shuffle=False))
      auroc = MultipleClassAUROC(
                        data=test,
                        class_names=class_names,
                        y_true = y_true
                        )
      m=Model(image_size=image_size, n_classes=len(class_names))
      model = m.get_model()

      model.load_weights(saved_weights_path)
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
      results = model.evaluate(test, callbacks=[auroc])


if __name__ == "__main__":
    main()