# coding:utf-8
import tensorflow as tf
import keras
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
import matplotlib.pyplot as plt

# 自作モジュール
import exp_func as ef

#コマンドライン入力用の変数
args = ef.argparser()

#出力ファイル名設定
history_name=args.HEADER+"_acc&loss_{0:%Y%m%d%H%M%S}.png".format(datetime.datetime.now())
result_name=args.HEADER+"_result_{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now())
analysis_name=args.HEADER+"_analysis_{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now())

def main():
    # GPU設定
    ef.GPU_configuration(args.GPU_USERATE, args.GPU_NUMBER)

    #Saving best trained parameter
    mc_cb = ModelCheckpoint(filepath = '{header}_best_weights_{time:%Y%m%d%H%M%S}.hdf5'.format(header=args.HEADER, time=datetime.datetime.now()), monitor = 'val_loss', save_best_only=True, save_weights_only=True, mode='auto', period=1)

    #データをロード
    if(args.LOAD == 'load_Img'):
        test_img_path, train_x, train_y, test_x, test_y = ef.load_Img(data_path=args.DATASET)

    elif(args.LOAD == 'load_AttnImg_CASNet'):
        test_img_path, train_x, train_y, test_x, test_y = ef.load_AttnImg_CASNet(data_path=args.DATASET)
        
        # メモリ開放
        keras.backend.clear_session()
        # 再度GPU設定
        ef.GPU_configuration(args.GPU_USERATE, args.GPU_NUMBER)

    if(args.MODEL == 'vgg16'):
        from models import vgg16 as vgg16
        model=vgg16.creat_VGG16_model()
    elif(args.MODEL == 'resnet50'):
        from models import resnet50 as res50
        model=res50.creat_ResNet50_model()

    #オプティマイザ
    # adam=optimizers.Adam(lr=0.001)
    sgd=optimizers.SGD(lr=0.001,momentum=0.9, decay=0.0005)
    
    model.compile(loss="mean_squared_error", optimizer=sgd, metrics=["accuracy"])

    #Training
    if args.SAVE_WEIGHTS == True:
        fit=model.fit(train_x, train_y, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE, callbacks = [mc_cb]) 
    else:
        fit=model.fit(train_x, train_y, nb_epoch=args.EPOCH, batch_size=args.BATCH, validation_split=args.VAL_RATE)

    #Saving history png file
    ef.save_histor(fit, history_name)

    #Saving trained parameter
    if args.SAVE_WEIGHTS == True:
        model.save_weights('{header}_weight_{time:%Y%m%d%H%M%S}.hdf5'.format(header=args.HEADER, time=datetime.datetime.now()))
    #Testing
    if(args.TEST == 'test_1img'):
        ef.test_1img(model, args.MODEL, test_img_path, test_x, test_y, result_name, analysis_name)
    elif(args.TEST == 'test_Img_AttnImg_Network'):
        ef.test_Img_AtnnImg_Network(model, args.TEST_IMG_CSV, args.TEST_SAL_CSV, result_name)

if __name__ == '__main__':
    main()