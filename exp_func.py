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
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm
import argparse
import datetime
import sys
import cv2

# 自作モジュール
from Tool import analyzer_function as af
from Tool import analyzer as alz
from Tool.casnet import get_saliency_map, create_CASNet

def GPU_configuration(gpu_userate, gpu_number):
    #GPU設定
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_userate,# GPU using rate
            visible_device_list=gpu_number, # GPU number
            allow_growth=True
        )
    )
    set_session(tf.Session(config=config))

# コマンドライン引数の設定
def argparser():
    parser = argparse.ArgumentParser()
    # GPU設定
    parser.add_argument("--GPU_USERATE", "-gu", type=float, help="input [0, 1]")
    parser.add_argument("--GPU_NUMBER", "-gn", type=str, help="input 0 or 1")

    # データ入力方法
    parser.add_argument("--DATASET", "-d", type=str, help="input a name of data csv file")

    #Output Name
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

def path2img(img_path,size=(224,224)):
    header = "../../dataset/Emotion6/"
    path = header + img_path
    image = load_img(path,target_size=size)#keras preprocess's function
    image = img_to_array(image)/255.#image-->ndarray
    return image

def normalize(sal):
    min = sal.min()
    max = sal.max()
    result = (sal-min)/(max-min)
    return result 

def path2attnimg(img_path, model, size=(224,224)):
    header = "../../dataset/Emotion6/"
    path = header + img_path
    image = load_img(path,target_size=size)#keras preprocess's function
    image = img_to_array(image)#image-->ndarray

    sal = get_saliency_map(path, model)
    sal = (normalize(sal) * 100).astype(int)
    sal = np.expand_dims(sal, axis=2)
    attnimg = image * sal /100. /255.

    return attnimg

def setup_dataset(dataset):
    img_path_list = np.array(dataset[:,0])
    label_list = np.array(dataset[:,1:8])

    print("Now loading Images and Labels...")
    img_list = np.array([path2img(img_path) for img_path in tqdm(img_path_list)])
    img_path_list = img_path_list[:,np.newaxis]

    return img_path_list, img_list, label_list

def setup_dataset_CASNet(dataset):
    #モデル生成
    model = create_CASNet()
    model.load_weights('Tool/salicon_generator_sigmoid_epoch_25.h5')

    img_path_list = np.array(dataset[:,0])
    label_list = np.array(dataset[:,1:8])

    print("Now loading Images and Labels...")
    img_list = np.array([path2attnimg(img_path, model) for img_path in tqdm(img_path_list)])
    img_path_list = img_path_list[:,np.newaxis]

    return img_path_list, img_list, label_list

def load_Img(data_path, test_split = 0.15):
    df_im=pd.read_csv(data_path,header=None)
    dataset=df_im.values

    if test_split == 1.0:
        img_path, data_x, data_y = setup_dataset(dataset)#train_img_pathは使わない
        return img_path, data_x, data_y
    else:
        train, test = train_test_split(dataset, test_size=test_split)

        train_img_path, train_x, train_y = setup_dataset(train)#train_img_pathは使わない
        test_img_path, test_x, test_y = setup_dataset(test)

        return test_img_path, train_x, train_y, test_x, test_y

def load_AttnImg_CASNet(data_path, test_split = 0.15):
    df_im=pd.read_csv(data_path,header=None)
    dataset=df_im.values

    if test_split == 1.0:
        img_path, data_x, data_y = setup_dataset_CASNet(dataset)#train_img_pathは使わない
        return img_path, data_x, data_y
    else:
        train, test = train_test_split(dataset, test_size=test_split)

        train_img_path, train_x, train_y = setup_dataset_CASNet(train)#train_img_pathは使わない
        test_img_path, test_x, test_y = setup_dataset_CASNet(test)

        return test_img_path, train_x, train_y, test_x, test_y

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

def test_1img(model, model_name, img_path_list, test_x, test_y, result_name, analysis_name):
    # 推定
    prediction_list=model.predict(test_x)
    result_list=np.concatenate([img_path_list,prediction_list],1)

    # 推定結果出力
    df = pd.DataFrame(result_list)
    df.to_csv(result_name,index=False, header=False)

    # 評価
    alz.basic_analisis(result_list, test_y, analysis_name)

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