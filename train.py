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

def main():
      config_file = "./config.ini"
      cp = ConfigParser()
      cp.read(config_file)
      output_dir = cp["DATA"].get("output_dir")
      if not os.path.exists(output_dir):
            os.makedirs(output_dir)
      dataset_name = cp["DATA"].get("dataset_name")
      dataset_val_name = cp["DATA"].get("dataset_val_name")
      csv_file = cp["DATA"].get("dataset_csv_file")
      csv_val_file = cp["DATA"].get("dataset_val_csv_file")
      img_per_shard = cp["DATA"].getint("img_per_shard")
      sharding = cp["DATA"].get("sharding")
      data_is_labeled = cp["DATA"].get("data_is_labeled")
      n_fold = cp["DATA"].getint("n_fold")
      image_size = cp["TRAIN"].getint("IMAGE_SIZE")
      batch_size = cp["TRAIN"].getint("batch_size")
      class_names = cp["TRAIN"].get("class_names").split(",")
      saved_model_path = cp['TRAIN'].get("saved_model_path")
    
      
      reader = Extract_TFrecord(image_size=image_size, batch_size=batch_size, sharding=sharding)
      reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=1, min_lr=0.00001)      
      m=Model(image_size=image_size, n_classes=len(class_names), imagenet=True)
      model = m.get_model()
      
      if n_fold > 1:
            fold_index = 0
            for fold in tqdm(range(n_fold)):
                  print(f"\n***********working on fold{str(fold_index+1)}************\n")
                  tf_record_dir_train = os.path.join(output_dir, 'fold'+str(fold+1), dataset_name+'_train')
                  tf_record_dir_val = os.path.join(output_dir, 'fold'+str(fold+1), dataset_name+'_val')
                  print(f"$$$$$$$$$$tf_record_dir_train:{tf_record_dir_train}")
                  print(f"$$$$$$$$$$tf_record_dir_val:{tf_record_dir_val}")
                  train = reader.get_dataset(augment=True, tf_record_dir=tf_record_dir_train, sharding=sharding)
                  valid = reader.get_dataset(augment=False, tf_record_dir=tf_record_dir_val, sharding=False, shuffle=False)
                  y_true = reader.get_test_label(reader.read_tfrecord(tf_record_dir_val, sharding=False, shuffle=False))
                  saved_model_fold = os.path.join(saved_model_path, 'saved_model', 'fold'+str(fold+1))
                  if not os.path.exists(saved_model_fold):
                        os.makedirs(saved_model_fold)
                  auroc = MultipleClassAUROC(
                        data=valid,
                        class_names=class_names,
                        saved_model_path = saved_model_fold,
                        y_true = y_true
                        )
                  train_counts, train_pos_counts = get_sample_counts(os.path.join(output_dir, 'fold'+str(fold+1), 'train.csv'))
                  class_weights = get_class_weights(
                                    train_counts,
                                    train_pos_counts,
                                    multiply=1
                                    )
                  # print(class_weights)
                  custom_loss = CustomLoss(class_names, class_weights)
                  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        loss = custom_loss.loss_fn,
                        metrics=['accuracy'])
                  history = model.fit(train, epochs=5, validation_data=valid, callbacks=[auroc, reduce_lr])
                  fold_index += 1
      else:
            tf_record_dir_train = os.path.join(output_dir, dataset_name+'_train')
            # print(f"tf_record_dir_train:{tf_record_dir_train}")
            tf_record_dir_val = os.path.join(output_dir, dataset_name+'_val') #chexpert_val_val
            train = reader.get_dataset(augment=True, tf_record_dir=tf_record_dir_train, sharding=sharding)
            valid = reader.get_dataset(augment=False, tf_record_dir=tf_record_dir_val, sharding=False, shuffle=False)            
            y_true = reader.get_test_label(reader.read_tfrecord(tf_record_dir_val, sharding=False, shuffle=False))
            saved_model_dir = os.path.join(saved_model_path, 'saved_model')
            if not os.path.exists(saved_model_dir):
                        os.makedirs(saved_model_dir)
            auroc = MultipleClassAUROC(
                        data=valid,
                        class_names=class_names,
                        y_true = y_true,
                        saved_model_path = os.path.join(saved_model_path, 'saved_model')
                        )
            train_counts, train_pos_counts = get_sample_counts(csv_file)
            class_weights = get_class_weights(
                              train_counts,
                              train_pos_counts,
                              multiply=1
                              )
            # print(class_weights)
            custom_loss = CustomLoss(class_names, class_weights)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  loss = custom_loss.loss_fn,
                  metrics=['accuracy'])
            history = model.fit(train, epochs=5, validation_data=valid, callbacks=[auroc, reduce_lr])
            # model.evaluate(train)
            

      # print(train)
      # for x in train.take(1):
      #       print(x)


if __name__ == "__main__":
    main()