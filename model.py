import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback

class Model:
      def __init__(self, image_size, n_classes, imagenet=True):
            self.image_size=image_size
            self.n_classes = n_classes
            self.inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
            if imagenet:
                  self.base_model = tf.keras.applications.DenseNet121(input_tensor=self.inputs,
                                                            include_top=False,
                                                            weights='imagenet',
                                                            pooling='avg')
            else:
                  self.base_model = tf.keras.applications.DenseNet121(input_tensor=self.inputs,
                                                            include_top=False,
                                                            weights=None,
                                                            pooling='avg')

      def get_model(self):
            x = self.base_model.output # set training = False to prevent batch_norm from updating
            predictions_new = tf.keras.layers.Dense(self.n_classes, name="predictions_new")(x)
            model = tf.keras.Model(self.inputs, predictions_new)
            return model

      def gradcam_model1(self):
            self.model1 = tf.keras.Model(inputs = self.inputs, outputs = self.base_model.get_layer("bn").output)
            self.model1.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
            return self.model1
      
      def gradcam_model2(self,category_index):
            def target_category_loss(x, category_index, nb_classes):
                  return tf.multiply(x, tf.one_hot([category_index], nb_classes))

            relu = self.base_model.get_layer("relu")
            avg_pool = self.base_model.get_layer("avg_pool")
            print(f'#####category:index:{category_index}#####')
            final_layer = tf.keras.layers.Lambda(lambda x: target_category_loss(x, category_index, self.n_classes))

            bn_inputs = tf.keras.Input(shape=self.model1.output.shape[1:4])
            x = relu(bn_inputs)
            x = avg_pool(x)
            predictions_new = tf.keras.layers.Dense(self.n_classes, activation=None, name="predictions_new")(x)
            output = final_layer(predictions_new)
            model2 = tf.keras.Model(inputs=bn_inputs, outputs = output)
            return model2



def main():
      m=Model(image_size=224, n_classes=14)
      model = m.get_model()
      print(model.summary())

if __name__ == "__main__":
    main()