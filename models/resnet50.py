# coding:utf-8

import os
import glob
import numpy as np
import cv2

import keras
import keras.backend as K
from keras import optimizers
from keras.engine.topology import Layer
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, Flatten, Dropout

from keras.models import Model
from collections import OrderedDict

def creat_ResNet50_model():
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
    x=Flatten()(resnet50.output)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(0.5)(x)
    output=Dense(7,activation='softmax')(x)
    new_model = Model(inputs=resnet50.inputs, outputs=output)
    return new_model

#only when executing "python3 resnet50.py"
if __name__  == "__main__":
    model=creat_ResNet50_model()
    model.summary()
