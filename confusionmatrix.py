import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import seaborn as sn

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, classnum, name="confusion_matrix"):
        super(ConfusionMatrix, self).__init__(name=name)
        self.classnum = classnum
        self.confusionmatrix = self.add_weight(shape = (classnum ,classnum), name='cm', initializer='zeros')

    def update_state(self, true, prediction):
        self.confusionmatrix.assign_add \
            (tf.math.confusion_matrix(true, prediction, dtype=tf.float32, num_classes=self.classnum))

    def result(self):
        return self.confusionmatrix

    def reset_state(self):
        self.confusionmatrix.assign(tf.zeros([self.classnum ,self.classnum]))

    def metric(self):
        cm = self.confusionmatrix
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        precision = t p /(t p +fp)
        recall = t p /(t p +fn)
        f1 = 2* precision * recall / (recall + precision)
        return precision, recall, f1


def pltmatrix(cm):
  plt.figure(figsize = (16,16))
  hmap = sn.heatmap(cm,annot=True)
  hmap.set_xlabel('Predict')
  hmap.set_ylabel('True')

def pltnormalmatrix(cm):
  plt.figure(figsize = (16,16))
  cm = cm/sum(cm)
  hmap = sn.heatmap(cm,annot=True, fmt=".1%")
  hmap.set_xlabel('Predict')
  hmap.set_ylabel('True')