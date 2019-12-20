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
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Multiply

from keras.models import Model
from keras.utils import plot_model
from collections import OrderedDict

#SEblock
def se_block(attn_img, img, channels):
    x = GlobalAveragePooling2D()(attn_img)
    x = Dense(channels // 16, activation='relu')(x)
    x = Dense((channels), activation='sigmoid')(x)
    return Multiply()([img, x])

def creat_aenet2way_model():
    vgg16_1 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))#input row images
    vgg16_2 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))#input attention images
    for layer in vgg16_2.layers:
        layer.name = layer.name + "_sec"
        layer.trainable = False

    x=Flatten()(vgg16_1.output)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(4096,activation='relu')(x)
    x=Dropout(0.5)(x)
    img_ch = x.shape.as_list()[1]#vgg16_1の出力マップのチャネル数を取得
    x=se_block(vgg16_2.output, x, img_ch)
    output=Dense(7,activation='softmax')(x)
    new_model = Model(inputs=[vgg16_1.input, vgg16_2.input], outputs=output)
    return new_model

#only when executing "python3 aenet_2way.py"
if __name__  == "__main__":
    model=creat_aenet2way_model()
    plot_model(model, to_file='aenet_2way.png')
    model.summary()
