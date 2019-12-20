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
import cv2

#自作モジュール
from Tool import analyzer_function as af
from Tool import analyzer as alz
from Tool import gradcam as gradcam

#コマンドライン引数の設定
def argparser():
    parser = argparse.ArgumentParser()
    #GPU setting
    parser.add_argument("--GPU_USERATE", "-gu", type=float, help="input [0, 1]")
    parser.add_argument("--GPU_NUMBER", "-gn", type=str, help="input 0 or 1")

    #Data
    parser.add_argument("--TRAIN_IMG_CSV", "-tri", type=str, help="input a name of data csv file")
    parser.add_argument("--TRAIN_SAL_CSV", "-trs", type=str, help="input a name of saliency csv file")
    parser.add_argument("--TEST_IMG_CSV", "-tei", type=str, help="input a name of test csv file")
    parser.add_argument("--TEST_SAL_CSV", "-tes", type=str, help="input a name of test csv file")

    #Output name
    parser.add_argument("--HEADER", "-he", type=str, help="input a header of result")

    #Training configuration
    parser.add_argument("--MODEL", "-m", type=str, help="input a name of model")
    parser.add_argument("--LOAD", "-l", type=str, help="input a name of load method")
    parser.add_argument("--TEST", "-t", type=str, help="input a name of test method")

    #Saving trained parameter
    parser.add_argument("--SAVE_WEIGHTS", "-sw", type=bool, help="input True or FAlse")

    #Making Grad-CAM images
    parser.add_argument("--GRADCAM", "-g", type=bool, help="input True or False")

    #Hyperparameter
    parser.add_argument("--EPOCH", "-e", type=int, help="input Epoch")
    parser.add_argument("--BATCH", "-b", type=int, help="input batch")
    parser.add_argument("--VAL_RATE", "-vr", type=float, help="input validation rate")
    return parser.parse_args()

def load_Img(data_path):
    df_im=pd.read_csv(data_path,header=None)
    df_im_array=df_im.values

    image_list=[]
    label_list=[]

    print("Now loading Images and Labels...")
    for x in tqdm(df_im_array):
        label=x[1:]
        label_list.append(label)
        filepath = x[0]
        image=load_img(filepath,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        image_list.append(image / 255.)

    # list to ndarray
    image_list = np.array(image_list)
    label_list = np.array(label_list)
    return image_list, label_list

def load_AttnImg_Network(data_path, sal_path):
    #loading dataset
    df_im=pd.read_csv(data_path,header=None)
    df_sal=pd.read_csv(sal_path,header=None)
    df_im_array=df_im.values
    df_sal_array=df_sal.values

    ##Dataset Section
    attn_image_list=[]
    label_list=[]

    print("Making attention images...")
    for x, y in zip(tqdm(df_im_array),tqdm(df_sal_array)):
        label=x[1:]
        label_list.append(label)
        filepath_im = x[0]
        filepath_sal = y[0]

        image=load_img(filepath_im,target_size=(224,224))#keras preprocess's function
        sal=load_img(filepath_sal, grayscale=True,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        sal=img_to_array(sal)/255.
        attn_img=image*sal
        attn_image_list.append(attn_img/255.)

    # list to ndarray
    attn_image_list = np.array(attn_image_list)
    label_list = np.array(label_list)
    return attn_image_list, label_list

def load_AttinImg_openCV(data_path):
    df_im=pd.read_csv(data_path,header=None)
    df_im_array=df_im.values

    attn_image_list=[]
    label_list=[]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    print("Now loading Images and Labels...")
    for x in tqdm(df_im_array):
        label=x[1:]
        label_list.append(label)
        filepath = x[0]
        image=load_img(filepath,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        sal=saliency.computeSaliency(image)
        sal=np.array(list(sal[1]))[ :,:,np.newaxis]#tuple->list->ndarray #newaxis for adding dimension
        attn_img=image*sal
        attn_image_list.append(attn_img / 255.)

    # list to ndarray
    attn_image_list = np.array(attn_image_list)
    label_list = np.array(label_list)
    return attn_image_list, label_list

def load_Img_AttnImg_Network(data_path, sal_path):
    #loading dataset
    df_im=pd.read_csv(data_path,header=None)
    df_sal=pd.read_csv(sal_path,header=None)
    df_im_array=df_im.values
    df_sal_array=df_sal.values

    img_list=[]
    attn_img_list=[]
    label_list=[]

    print("Loading images and Making attention images...")
    for x, y in zip(tqdm(df_im_array,df_sal_array)):
        label=x[1:]
        label_list.append(label)

        filepath_im = x[0]
        filepath_sal = y[0]
        image=load_img(filepath_im,target_size=(224,224))#keras preprocess's function
        sal=load_img(filepath_sal, grayscale=True,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        img_list.append(image)
        sal=img_to_array(sal)/255.
        attn_img=image*sal
        attn_img_list.append(attn_img/255.)

    # list to ndarray
    img_list = np.array(img_list)
    attn_img_list = np.array(attn_img_list)
    label_list = np.array(label_list)

    return img_list, attn_img_list, label_list

def save_histor(fit, history_name):
    #Visualizing loss&acc
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

    # Plot the loss in the history
    axR.plot(fit.history['acc'],label="acc for training")
    axR.plot(fit.history['val_acc'],label="acc for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')
    
    fig.savefig(history_name)
    plt.close()

def test_Img(model, model_name, img_csv, result_name, gcam, header):
    df_im=pd.read_csv(img_csv,header=None)
    df_im_array=df_im.values

    data_list=[]
    image_testlist=[]

    #make gradcam directory
    if gcam == True:
        gcam_dir="{header}/gradcam_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
        os.mkdir(gcam_dir)
        hmap_dir="{header}/heatmap_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
        os.mkdir(hmap_dir)

    for x in tqdm(df_im_array):
        filepath = x[0]
        #filepath & outputs --> lists
        data_list.append(np.array([filepath]))
        image=load_img(filepath,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        image_testlist.append(image/255.)
        
        if gcam == True:
            gradcam.get_gradcam(filepath, model_name, model, gcam_dir)
            gradcam.get_heatmap(filepath, model_name, model, hmap_dir)


    image_testlist=np.array(image_testlist)
    # prediction
    result_list=model.predict(image_testlist)
    result_list=np.concatenate([data_list,result_list],1)
    df = pd.DataFrame(result_list)
    df.to_csv(result_name,index=False, header=False)

def test_AttnImg_Network(model, model_name, img_csv, sal_csv, result_name, gcam, header):
    df_im=pd.read_csv(img_csv,header=None)
    df_im_array=df_im.values
    df_sal=pd.read_csv(sal_csv,header=None)
    df_sal_array=df_sal.values

    data_list=[]
    attn_image_testlist=[]

    #make gradcam directory
    if gcam == True:
        gcam_dir="{header}/gradcam_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
        os.mkdir(gcam_dir)
        hmap_dir="{header}/heatmap_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
        os.mkdir(hmap_dir)

    for x,y in zip(tqdm(df_im_array), tqdm(df_sal_array)):
        filepath_im = x[0]
        filepath_sal = y[0]

        data_list.append(np.array([filepath_im]))
        image=load_img(filepath_im,target_size=(224,224))#keras preprocess's function
        sal=load_img(filepath_sal, grayscale=True,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        sal=img_to_array(sal)/255.
        attn_img=image*sal
        attn_image_testlist.append(attn_img/255.)

        if gcam == True:
            gradcam.get_gc_hm_attnimg(filepath_im, attn_img, model_name, model, gcam_dir, hmap_dir)

    attn_image_testlist=np.array(attn_image_testlist)
    #prediction
    result_list = model.predict(attn_image_testlist)
    result_list=np.concatenate([data_list,result_list],1)

    df = pd.DataFrame(result_list)
    df.to_csv(result_name,index=False, header=False)

def test_AttnImg_openCV(model, model_name, img_csv, result_name, gcam, header):
    df_im=pd.read_csv(img_csv,header=None)
    df_im_array=df_im.values

    data_list=[]
    attn_image_testlist=[]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    #make gradcam directory
    if gcam == True:
        gcam_dir="{header}/gradcam_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
        os.mkdir(gcam_dir)
        hmap_dir="{header}/heatmap_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
        os.mkdir(hmap_dir)

    for x in tqdm(df_im_array):
        label=x[1:]
        data_list.append(label)
        filepath = x[0]
        image=load_img(filepath,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        sal=saliency.computeSaliency(image)
        sal=np.array(list(sal[1]))[ :,:,np.newaxis]#tuple->list->ndarray #newaxis for adding dimension
        attn_img=image*sal/255.
        attn_image_testlist.append(attn_img)
        
        if gcam == True:
            gradcam.get_gc_hm_attnimg(filepath, attn_img, model_name, model, gcam_dir, hmap_dir)

    attn_image_testlist=np.array(attn_image_testlist)
    # prediction
    result_list=model.predict(attn_image_testlist)
    result_list=np.concatenate([data_list,result_list],1)
    df = pd.DataFrame(result_list)
    df.to_csv(result_name,index=False, header=False)

def test_Img_AtnnImg_Network(model, img_csv, sal_csv, result_name):
    df_im=pd.read_csv(img_csv,header=None)
    df_im_array=df_im.values
    df_sal=pd.read_csv(sal_csv,header=None)
    df_sal_array=df_sal.values

    data_list=[]
    img_testlist=[]
    attn_img_testlist=[]

    for x, y in zip(tqdm(df_im_array,df_sal_array)):
        filepath_im = x[0]
        filepath_sal = y[0]

        data_list.append(np.array([filepath_im]))
        image=load_img(filepath_im,target_size=(224,224))#keras preprocess's function
        image=img_to_array(image)#image-->ndarray
        img_testlist.append(image/255.)

        sal=load_img(filepath_sal, grayscale=True,target_size=(224,224))#keras preprocess's function
        sal=img_to_array(sal)/255.
        attn_img=image*sal
        attn_img_testlist.append(attn_img/255.)

    img_testlist=np.array(img_testlist)
    attn_img_testlist=np.array(attn_img_testlist)
    
    #prediction
    result_list = model.predict([img_testlist,attn_img_testlist])
    result_list=np.concatenate([data_list,result_list],1)

    df = pd.DataFrame(result_list)
    df.to_csv(result_name,index=False, header=False)