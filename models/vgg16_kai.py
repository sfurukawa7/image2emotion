# coding:utf-8
import os
import glob
import numpy as np
import cv2

import keras
import keras.backend as K
from keras import optimizers
from keras.engine.topology import Layer
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Average

from keras.models import Model
from collections import OrderedDict
from keras.utils import plot_model

def creat_VGG16_model():
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

    #branch1
    x=Conv2D(128, (1, 1), padding='same', activation='relu',name='branch1_bottleneck')(vgg16.get_layer('block1_pool').output)
    x=GlobalAveragePooling2D()(x)
    x=Dense(4096, activation='relu', name='branch1_fc1')(x)
    x=Dropout(0.5)(x)
    output1=Dense(7, activation='relu', name='branch1_fc2')(x)

    #branch2
    x=Conv2D(128, (1, 1), padding='same', activation='relu',name='branch2_bottleneck')(vgg16.get_layer('block2_pool').output)
    x=GlobalAveragePooling2D()(x)
    x=Dense(4096, activation='relu', name='branch2_fc1')(x)
    x=Dropout(0.5)(x)
    output2=Dense(7, activation='relu', name='branch2_fc2')(x)

    #branch3
    x=Conv2D(128, (1, 1), padding='same', activation='relu',name='branch3_bottleneck')(vgg16.get_layer('block3_pool').output)
    x=GlobalAveragePooling2D()(x)
    x=Dense(4096, activation='relu', name='branch3_fc1')(x)
    x=Dropout(0.5)(x)
    output3=Dense(7, activation='relu', name='branch3_fc2')(x)

    #branch4
    x=Conv2D(128, (1, 1), padding='same', activation='relu',name='branch4_bottleneck')(vgg16.get_layer('block4_pool').output)
    x=GlobalAveragePooling2D()(x)
    x=Dense(4096, activation='relu', name='branch4_fc1')(x)
    x=Dropout(0.5)(x)
    output4=Dense(7, activation='relu', name='branch4_fc2')(x)

    #branch5
    x=Conv2D(128, (1, 1), padding='same', activation='relu',name='branch5_bottleneck')(vgg16.get_layer('block5_pool').output)
    x=GlobalAveragePooling2D()(x)
    x=Dense(4096, activation='relu', name='branch5_fc1')(x)
    x=Dropout(0.5)(x)
    output5=Dense(7, activation='relu', name='branch5_fc2')(x)

    #Fuison
    x=Average()([output1, output2, output3, output4, output5])
    output=Dense(7,activation='softmax')(x)
    new_model = Model(inputs=vgg16.inputs, outputs=output)
    return new_model

#only when executing "python3 VGG16_v4.py"
if __name__  == "__main__":
    model=creat_VGG16_model()
    model.summary()
    plot_model(model, to_file='vgg16_v4.png')
    model.save('vgg16_v4.h5', include_optimizer=False)