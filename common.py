"""
    import common library
"""
import os
import ast
import datetime as dt
import argparse

# slack nofify
from slackclient import SlackClient

# matplot and graph
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns

# as usual
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

# image processing and augmentation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import albumentations
import cv2

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,
    Resize, Normalize, Rotate, RandomCrop, Crop, CenterCrop, DualTransform
)

# tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

###################################
# limit GPU when training data
from keras.backend.tensorflow_backend import set_session
# import horovod.keras as hvd
# # horovod: initialize Horovod.
# hvd.init()
# tensorFlow wizardry
config = tf.ConfigProto()
# don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# device list
# config.gpu_options.visible_device_list = str(hvd.local_rank())
# create a session with the above options specified.
set_session(tf.Session(config=config))
###################################

# root_path = '/home/trinhnh1/Documents/train_data/kaggle/quick_draw/'
root_path = './'

start = dt.datetime.now()