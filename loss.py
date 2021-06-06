import tensorflow as tf

class CustomLoss():
      def __init__(self, class_names, class_weights):
            self.class_names = class_names
            self.class_weights = class_weights

      def loss_fn(self, y_true,y_pred):
            loss = 0
            y_pred = tf.keras.activations.sigmoid(y_pred)
            y_pred = tf.clip_by_value(y_pred, 0.0000001, 1-0.0000001)
            for i in range(len(self.class_names)):
                  loss -= (self.class_weights[i][1]*tf.transpose(y_true)[i]*tf.math.log(tf.transpose(y_pred)[i]) + self.class_weights[i][0]*(1-tf.transpose(y_true)[i])*tf.math.log(1-tf.transpose(y_pred)[i]))
            return loss/len(self.class_names)