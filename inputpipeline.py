import pandas as pd
import glob
import numpy as np
import tensorflow as tf



def takedata(label, useridfirst, useridfinal):
    dataset = pd.DataFrame()

    for id in range(useridfirst, useridfinal + 1):
        if id < 10:
            userid = 'user' + '0' + str(id)
        else:
            userid = 'user' + str(id)
        path = "./HAR/RawData/" + '*' + userid + '*'

        accpath = []
        gyropath = []
        for file in glob.glob(path):
            if 'acc' in file:
                accpath.append(file)
            if 'gyro' in file:
                gyropath.append(file)
        print(accpath)
        print(gyropath)

        for pth in accpath:
            apath = pth
            gpath = apath.replace('acc', 'gyro')
            acc = pd.read_table(apath, delimiter=' ', names=["acc_x", "acc_y", "acc_z"])
            gyro = pd.read_table(gpath, delimiter=' ', names=["gyro_x", "gyro_y", "gyro_z"])
            userdata = pd.concat([acc, gyro], axis=1)
            print(gyro, gpath)

            n = acc.shape[0]
            np_label = np.zeros(n)
            position = label[(label["experiment_id"] == int(apath[-13:-11])) & (label["user_id"] == int(apath[-6:-4]))]
            for index, row in position.iterrows():
                np_label[row['start_position'] - 1:row['end_position']] = row['activity_id']
            df_label = pd.DataFrame(np_label, columns=['label'])

            alldata = pd.concat([userdata, df_label], axis=1)
            alldata = alldata.drop(alldata[alldata['label'] == 0].index)

            dataset = pd.concat([dataset, alldata], ignore_index=True)
    return dataset


def zscore(data):
    dataz = data
    for name, _ in dataz.items():
        if name == 'label':
            continue
        dataz[name] = (dataz[name] - dataz[name].mean()) / dataz[name].std()

    return dataz

def tfdataset(dataname,ztraindata,ztestdata,window,shift,batchsize):

    training_dataset = tf.data.Dataset.from_tensor_slices((ztraindata[dataname].values, ztraindata['label'].values - 1))
    test_dataset = tf.data.Dataset.from_tensor_slices((ztestdata[dataname].values, ztestdata['label'].values - 1))
    dstrain = training_dataset.window(size=window, shift=shift, drop_remainder=True)
    dstrain = dstrain.flat_map(lambda x, y: tf.data.Dataset.zip((x, y)))
    dstrain = dstrain.batch(window)
    dstest = test_dataset.batch(window, drop_remainder=True)
    dstrain = dstrain.repeat(-1)
    dstrain = dstrain.batch(batchsize)
    dstest = dstest.batch(1)

    return dstrain,dstest