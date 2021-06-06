from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import tensorflow as tf
from read_tfrecord import Extract_TFrecord
import numpy as np
import os
import shutil

class MultipleClassAUROC(Callback):
    def __init__(self, data, class_names, y_true, saved_model_path=None):
        super().__init__()
        self.stats = {"best_mean_auroc": 0}
        self.data = data
        self.class_names = class_names
        self.y_true = y_true
        self.checkpoint_path = saved_model_path
        self.aurocs = {}
        # self.reader = Extract_TFrecord()

    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.

        """
        print("\n*********************************")
        # self.checkpoint_path = "saved_model/chexpert_epoch"+str(epoch)+".h5"
        """
        y_hat shape: (#samples, len(class_names))
        y: [(#samples, 1), (#samples, 1) ... (#samples, 1)]
        """
        y_hat = tf.keras.activations.sigmoid(self.model.predict(self.data))
        y = [y.numpy() for y in self.y_true][0]
        print(f"*** epoch#{epoch + 1} dev auroc ***")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]] = score
            current_auroc.append(score)
            print(f"{i+1}. {self.class_names[i]}: {score}")
        print("*********************************")

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print(f"mean auroc: {mean_auroc}")
        if mean_auroc > self.stats["best_mean_auroc"]:
            
            print(f"update best auroc from {self.stats['best_mean_auroc']} to {mean_auroc}")
            checkpoint_path = os.path.join(self.checkpoint_path, 'epoch_'+str(epoch+1)+ '.h5')
            best_weights_path = os.path.join(self.checkpoint_path, 'best_weights.h5')
            print(f"save model weights to {checkpoint_path}")
            print(f"update best weights to {best_weights_path}")
            self.model.save_weights(checkpoint_path)
            shutil.copy(checkpoint_path, best_weights_path)
            self.stats['best_mean_auroc'] = mean_auroc
            print("*********************************")
        return

    def on_test_end(self, logs=None):
        y_hat = tf.keras.activations.sigmoid(self.model.predict(self.data))
        y = [y.numpy() for y in self.y_true][0]
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]] = score
            current_auroc.append(score)
            print(f"{i+1}. {self.class_names[i]}: {score}")
        print("*********************************")

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print(f"mean auroc: {mean_auroc}")
    