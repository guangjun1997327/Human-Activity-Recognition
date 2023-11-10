import gin
import pandas as pd
import glob
import numpy as np
import logging
import tensorflow as tf
from absl import app, flags
from inputpipeline import takedata, zscore, tfdataset
from train import Trainer
from evaluation import evaluate

from rnn_model import rnn

def main(argv):
    label = pd.read_table("./HAR/RawData/labels.txt",  delimiter=' ', names=["experiment_id", "user_id", "activity_id", "start_position", "end_position"])
    traindata = takedata(label, 1, 21)
    testdata = takedata(label, 22, 27)
    ztraindata = zscore(traindata)
    ztestdata = zscore(testdata)

    dataname = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']
    dstrain, dstest = tfdataset(dataname,ztraindata,ztestdata,250,125,32)

    model100 = rnn((250,6),256,3,200,0.2)


    if FLAGS.mode == 'train':

        trainer = Trainer(model100, dstrain, dstest, 10000, 100, 0.001, 'softmax','/content/drive/MyDrive/Colab Notebooks/HAR/checkpoint')
        for _ in trainer.train():
            continue
    else:
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model100)
        evaluate(model100, dstest, 12, checkpoint, '/content/drive/MyDrive/Colab Notebooks/HAR/checkpoint/ckpt-5')

if __name__ == "__main__":
    app.run(main)