# coding:utf-8
import os
import glob
import numpy as np
import cv2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import keras.backend as K
from keras import optimizers
from keras.engine.topology import Layer
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Flatten, Dropout

from keras.models import Model
from collections import OrderedDict

def creat_VGG16_model():
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    x=Flatten()(vgg16.output)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(0.5)(x)
    output=Dense(7,activation='softmax')(x)
    new_model = Model(inputs=vgg16.inputs, outputs=output)
    return new_model

#only when executing "python3 VGG16_ft_v3.py"
if __name__  == "__main__":
    model=creat_VGG16_model()
    model.summary()