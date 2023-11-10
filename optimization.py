import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import wandb
from wandb import init, log, join

wandb.login()


def main():
    run = wandb.init(project='sweep53')

    total_step = wandb.config.total_step
    drop_rate = wandb.config.drop_rate
    # model parameter
    dense = wandb.config.dense
    layers = wandb.config.layers
    lstm = wandb.config.lstm
    # model
    model1 = rnn((250, 6), lstm, layers, dense, drop_rate)
    trainer = Trainer(model1, dstrain, dstest, 10000, 100, 0.001, 'softmax',
                      '/content/drive/MyDrive/Colab Notebooks/HAR/checkpoint',
                      '/content/drive/MyDrive/Colab Notebooks/HAR/ck2')
    for _ in trainer.train():
        continue


sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
        {
            'total_step': {'values': [10000]},
            'dense': {'values': [20, 100, 200]},
            'drop_rate': {'values': [0, 0.2]},
            'layers': {'values': [1, 2, 3]},
            'lstm': {'values': [128, 256]}

        }
}

sweep_7 = wandb.sweep(sweep=sweep_configuration, project='sweep53')
wandb.agent(sweep_7, function=main, count=70)