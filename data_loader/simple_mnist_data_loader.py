from base.base_data_loader import BaseDataLoader
from keras.applications.mobilenet import preprocess_input
from keras.datasets import mnist
import pandas as pd
import os
import cv2
import ast
import numpy as np
from tensorflow import keras

root_path = '/home/trinhnh1/Documents/train_data/kaggle/quick_draw/'
SHUFFLE_DATA = os.path.join(root_path,'input/shuffle_data/')
NCSVS = 100
BASE_SIZE = 256
size = 64
def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(SHUFFLE_DATA, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 1)) # for MobileNet
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=340)
                yield x, y

class SimpleMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleMnistDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape((-1, 28 * 28))
        self.X_test = self.X_test.reshape((-1, 28 * 28))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test


class MobileNetDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MobileNetDataLoader, self).__init__(config)
        test_df = pd.read_csv(os.path.join(SHUFFLE_DATA, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
        self.X_test = df_to_image_array_xd(test_df, size)
        self.y_test = keras.utils.to_categorical(test_df.y, num_classes=340)
        self.train_datagen = image_generator_xd(size=size, batchsize=256, ks=range(NCSVS - 1))

    def get_train_data(self):
        # return self.train_datagen.x, self.train_datagen.y
        return self.X_test, self.y_test

    def get_test_data(self):
        return self.X_test, self.y_test
