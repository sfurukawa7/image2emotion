# coding:utf-8
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import models, optimizers
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import trange, tqdm
import argparse
import datetime
import sys

#自作モジュール
import exp_function as ef
from Tool import analyzer_function as af
from Tool import analyzer as alz

#コマンドライン引数のグローバル変数
args = ef.argparser()

#出力ファイル名設定
history_name=args.HEADER+"_acc&loss_{0:%Y%m%d%H%M%S}.png".format(datetime.datetime.now())
result_name=args.HEADER+"_result_{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now())
analysis_name=args.HEADER+"_analysis_{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now())

def main():
    #GPU Configuration
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=args.GPU_USERATE,# GPU using rate
            visible_device_list=args.GPU_NUMBER, # GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))

    if(args.MODEL == 'vgg16'):
        from models import vgg16 as vgg16
        model=vgg16.creat_VGG16_model()
    elif(args.MODEL == 'aenet2way'):
        from models import aenet_2way as ae2
        model=ae2.creat_aenet2way_model()
    elif(args.MODEL == 'resnet50'):
        from models import resnet50 as res50
        model=res50.creat_ResNet50_model()
        
    #Optimizer
    # adam=optimizers.Adam(lr=0.001)
    sgd=optimizers.SGD(lr=0.001,momentum=0.9, decay=0.0005)
    #Compling
    model.compile(loss="mean_squared_error", optimizer=sgd, metrics=["accuracy"])
    
    #Saving best trained parameter
    mc_cb = ModelCheckpoint(filepath = '{header}_best_weights_{time:%Y%m%d%H%M%S}.hdf5'.format(header=args.HEADER, time=datetime.datetime.now()), monitor = 'val_loss', save_best_only=True, save_weights_only=True, mode='auto', period=1)

    #Loading dataset
    if(args.LOAD == 'load_Img'):
        image_list, label_list = ef.load_Img(args.TRAIN_IMG_CSV)
        #Training
        if args.SAVE_WEIGHTS == True:
            fit=model.fit(image_list, label_list, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE, callbacks = [mc_cb]) 
        else:
            fit=model.fit(image_list, label_list, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE)

    elif(args.LOAD == 'load_AttnImg_Network'):
        attn_image_list, label_list = ef.load_AttnImg_Network(args.TRAIN_IMG_CSV, args.TRAIN_SAL_CSV)
        #Saving trained parameter
        if args.SAVE_WEIGHTS == True:
            fit=model.fit(attn_image_list, label_list, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE, callbacks = [mc_cb])
        else:
            fit=model.fit(attn_image_list, label_list, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE)

    elif(args.LOAD == 'load_Img_AttnImg_Network'):
        img_list, attn_img_list, label_list = ef.load_Img_AttnImg_Network(args.TRAIN_IMG_CSV, args.TRAIN_SAL_CSV)
        #Training
        if args.SAVE_WEIGHTS == True:
            fit=model.fit([img_list, attn_img_list], label_list, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE, callbacks = [mc_cb])
        else:
            fit=model.fit([img_list, attn_img_list], label_list, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE)

    #Saving history png file
    ef.save_histor(fit, history_name)

    #Saving trained parameter
    if args.SAVE_WEIGHTS == True:
        model.save_weights('{header}_weight_{time:%Y%m%d%H%M%S}.hdf5'.format(header=args.HEADER, time=datetime.datetime.now()))

    #Testing
    if(args.TEST == 'test_Img'):
        ef.test_Img(model, args.MODEL, args.TEST_IMG_CSV, result_name, args.GRADCAM, args.HEADER)
    elif(args.TEST == 'test_AttnImg_Network'):
        ef.test_AttnImg_Network(model, args.MODEL, args.TEST_IMG_CSV, args.TEST_SAL_CSV, result_name, args.GRADCAM, args.HEADER)
    elif(args.TEST == 'test_Img_AttnImg_Network'):
        ef.test_Img_AtnnImg_Network(model, args.TEST_IMG_CSV, args.TEST_SAL_CSV, result_name)

    #Evaluation Section
    alz.one_analisis(result_name, args.TEST_IMG_CSV, analysis_name)

if __name__ == '__main__':
    main()