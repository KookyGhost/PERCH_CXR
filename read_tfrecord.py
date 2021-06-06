import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from configparser import ConfigParser

class Extract_TFrecord():
      def __init__(self, image_size=None, batch_size=None, sharding=None, grad_cam=False):
            self.image_size = image_size
            self.batch_size = batch_size
            self.AUTO = tf.data.experimental.AUTOTUNE
            self.sharding = sharding
            self.grad_cam = grad_cam

      def read_tfrecord(self, tf_record_dir, sharding, shuffle=True):
            if sharding == True:
                  tfrecords_pattern = tf_record_dir + '_*-of-*.records'
                  print(f'tfrecord_pattern:{tfrecords_pattern}')
                  files = tf.io.matching_files(tfrecords_pattern)
                  files = tf.random.shuffle(files)
                  shards = tf.data.Dataset.from_tensor_slices(files)
                  dataset = shards.interleave(tf.data.TFRecordDataset,num_parallel_calls=20,cycle_length=20)
                  
            else:
                  dataset = tf.data.TFRecordDataset(tf_record_dir+'.records')
            if shuffle == True:
                  dataset = dataset.shuffle(buffer_size=2000)
            print(f'**********size of dataset:{self.get_size(dataset)}**********')
            # dataset = dataset.cache(cache_dir)
            return dataset
      

      def parse_tfrecord(self, example):
            features = {
                  'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                  'image_raw': tf.io.FixedLenFeature([], tf.string),  
                  }

            features_nolabel = {
                  'image_raw': tf.io.FixedLenFeature([], tf.string),  
                  }
            try: 
                  parsed_features = tf.io.parse_single_example(example, features) 
            except:
                  parsed_features = tf.io.parse_single_example(example, features_nolabel)

            decoded_image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=3)
            resized_image = tf.image.resize(decoded_image, size=(self.image_size,self.image_size))
            if self.grad_cam:
                  return resized_image 
            try:
                  return resized_image, parsed_features['label']
            except:
                  return resized_image
      
      def parse_tflabel(self, example):
            features = {
            'label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'image_raw': tf.io.FixedLenFeature([], tf.string),   
            }
            parsed_features = tf.io.parse_single_example(example, features)
            return parsed_features['label']
      
      def get_size(self, raw_dataset):
            return len(list(raw_dataset))
                  
      def get_test_label(self, dataset):
            decoded = dataset.map(self.parse_tflabel, num_parallel_calls=self.AUTO)
            return decoded.batch(self.get_size(dataset))

      def data_augment(self, decoded_image, label):
            image = tf.image.random_flip_left_right(decoded_image)
            # image = tf.image.resize_with_crop_or_pad(image, self.image_size + 6, self.image_size + 6)
            # image = tf.image.random_crop(image, size=[self.image_size, self.image_size, 3])
            image = tf.image.random_brightness(image, max_delta=0.1) 
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.clip_by_value(image, 0, 255)
            return image, label

      def data_preprocess(self, image, label=None):
            image = tf.cast(image, tf.float16, name=None)
            image = tf.keras.applications.densenet.preprocess_input(image)
            if label is not None:
                  return image, tf.cast(label, tf.float32)
            else:
                  return image

      def get_dataset(self, tf_record_dir, augment=True, shuffle=True, sharding=True):
            print(f'*******tf_recrod_dir*********:{tf_record_dir}')
            # print(f'********sharding*********: {self.sharding}')
            if shuffle == True:
                  if sharding == True:
                        dataset = self.read_tfrecord(tf_record_dir, sharding=True)
                  else: 
                        dataset = self.read_tfrecord(tf_record_dir, sharding=False)
            else:
                  if sharding == True:
                        dataset = self.read_tfrecord(tf_record_dir, sharding=True, shuffle=False)
                  else:
                        dataset = self.read_tfrecord(tf_record_dir, sharding=False, shuffle=False)
            dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=self.AUTO)
            if augment == True:
                  dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO)
            dataset = dataset.map(self.data_preprocess, num_parallel_calls=self.AUTO)
            print(f'dataset batch_size:{self.batch_size}')
            return dataset.batch(self.batch_size).prefetch(self.AUTO) 

      def _deprocess_numpy_input(self, x):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            x[..., 0] *= std[0]
            x[..., 1] *= std[1]
            x[..., 2] *= std[2]
            x[..., 0] += mean[0]
            x[..., 1] += mean[1]
            x[..., 2] += mean[2]
            x *= 255.
            return x   
    
     

def main():
      config_file = "./config.ini"
      cp = ConfigParser()
      cp.read(config_file)
      output_dir = cp["DATA"].get("output_dir")
      if not os.path.exists(output_dir):
            os.makedirs(output_dir)
      dataset_name = cp["DATA"].get("dataset_name")
      csv_file = cp["DATA"].get("dataset_csv_file")
      img_per_shard = cp["DATA"].getint("img_per_shard")
      sharding = cp["DATA"].get("sharding")
      n_fold = cp["DATA"].getint("n_fold")
      IMAGE_SIZE = cp["TRAIN"].getint("IMAGE_SIZE")
      batch_size = cp["TRAIN"].getint("batch_size")

      tf_record_dir = os.path.join(output_dir, dataset_name)
      reader = Extract_TFrecord(image_size=IMAGE_SIZE, batch_size=batch_size, sharding=sharding)
      # dataset = reader.read_tfrecord(tf_record_dir, sharding=sharding)
      # get_dataset(self, tf_record_dir, training=True)
      dataset = reader.get_dataset(tf_record_dir=tf_record_dir, augment=True)
      print(dataset)
      print(f'tf_record_dir:{tf_record_dir}')
      for x in dataset.take(1):
            print(x)




if __name__ == "__main__":
    main()