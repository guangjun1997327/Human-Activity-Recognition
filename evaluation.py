import tensorflow as tf
import logging
from confusionmatrix import ConfusionMatrix, pltmatrix,pltnormalmatrix

def evaluate(model, ds_test, classnum, checkpoint,checkpath):
    checkpoint.restore(checkpath)
    cm = ConfusionMatrix(classnum)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    for image, label in ds_test:
        pred = model(image, training=False)
        test_accuracy(label, pred)
        pred = tf.math.argmax(pred, -1)
        label = tf.squeeze(label)
        pred = tf.squeeze(pred)
        cm.update_state(label, pred)

    # template = 'Test Loss: {}, Test Accuracy: {}'
    # logging.info(template.format(loss, accuracy * 100))

    template = 'Confusion Matrix: {}'
    logging.info(template.format(cm.result().numpy()))
    precision, recall, f1 = cm.metric()
    template = 'Precision: {}, Recall: {}, F1: {}, Accuracy:{}'
    logging.info(template.format(precision, recall, f1, test_accuracy.result() * 100))
    pltmatrix(cm.result())
    pltnormalmatrix(cm.result())