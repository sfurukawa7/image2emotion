import pandas as pd
import numpy as np
import datetime
import sys
from tqdm import trange, tqdm

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import load_img, img_to_array

from Tool import analyzer_function as af

#1回の結果で分析
#入力は「推定ファイル名」と「正解ファイル名」
def one_analisis(pre_csv,gt_csv,analysis_name):
    df_pre = pd.read_csv(pre_csv, header=None)
    df_gt = pd.read_csv(gt_csv, header=None)
    analisis=[]
    analisis.append(["image","Kullback-Leibler divergence","Chebyshev distance","Canberra metric","Classification"])
    pre=df_pre.values
    gt=df_gt.values

    #画像の名前リスト
    name_list=pre[:,0]

    #name_label部分を削除&型を指定
    #推定
    pre=np.delete(pre,0,1)
    pre=pre.astype(np.float64)
    #正解
    gt=np.delete(gt,0,1)
    gt=gt.astype(np.float64)
    
    #各画像での指標を計算
    for x,y,z in zip(pre,gt,name_list):
        data=[z,af.KL(y,x),af.CD(y,x),af.CM(y,x),af.Class(y,x)]
        analisis.append(data)

    #list -> ndarray    
    analisis_value=np.array(analisis)
    
    #各指標の平均を算出
    #ここで要素の型を指定している
    avg_KL=np.mean(analisis_value[1:,1].astype(np.float64))
    avg_CD=np.mean(analisis_value[1:,2].astype(np.float64))
    avg_CM=np.mean(analisis_value[1:,3].astype(np.float64))
    avg_Class=np.sum(analisis_value[1:,4].astype(np.float64))/name_list.size*100
    analisis.append(["Average",avg_KL, avg_CD, avg_CM, avg_Class])
    df_analisis = pd.DataFrame(analisis)

    df_analisis.to_csv(analysis_name,index=False, header=False)

# 1回の結果で分析
#入力は「推定ファイル名」と「正解ファイル名」
def basic_analisis(result_list, test_y, analysis_name):
    analisis=[]
    analisis.append(["image","Kullback-Leibler divergence","Chebyshev distance","Canberra metric","Classification"])

    #画像の名前リスト
    name_list=result_list[:,0]

    #name_label部分を削除&型を指定
    #推定
    
    result_list=np.delete(result_list,0,1)
    result_list=result_list.astype(np.float64)
    #正解
    test_y=test_y.astype(np.float64)
    
    #各画像での指標を計算
    for x,y,z in zip(result_list,test_y,name_list):
        data=[z,af.KL(y,x),af.CD(y,x),af.CM(y,x),af.Class(y,x)]
        analisis.append(data)

    #list -> ndarray    
    analisis_value=np.array(analisis)
    
    #各指標の平均を算出
    #ここで要素の型を指定している
    avg_KL=np.mean(analisis_value[1:,1].astype(np.float64))
    avg_CD=np.mean(analisis_value[1:,2].astype(np.float64))
    avg_CM=np.mean(analisis_value[1:,3].astype(np.float64))
    avg_Class=np.sum(analisis_value[1:,4].astype(np.float64))/name_list.size*100
    analisis.append(["Average",avg_KL, avg_CD, avg_CM, avg_Class])
    df_analisis = pd.DataFrame(analisis)

    df_analisis.to_csv(analysis_name,index=False, header=False)

#5回の分析結果の平均を算出
def five_analysis():

    avg = 0
    for i in range(5):
        print("{}st analisis csv:".format(i+1))
        a=input()
        df_a = pd.read_csv(a, header=None)
        a_value=df_a.values
        
        if i == 0:
            #headerと画像リストを抽出
            header=a_value[0]
            #headerはデフォで横ベクトルなので縦ベクトルにする
            name_list=a_value[1:,0].reshape((-1, 1))

        #データ部分
        a_value=a_value[1:,1:].astype(np.float64)

        avg += a_value

    avg = avg/5
    avg = np.hstack((name_list,avg))
    avg = np.vstack((header,avg))
    df_avg = pd.DataFrame(avg.tolist())

    #モデルの名前を入力
    print("model name:")
    name_header = input()
    name_output="V_{name}_{time:%Y%m%d%H%M%S}.csv".format(name=name_header, time=datetime.datetime.now())
    df_avg.to_csv(name_output,index=False, header=False)

#正解した画像のpathをcsvで出力
def correct_image(pre_csv,gt_csv):
    df_pre = pd.read_csv(pre_csv, header=None)
    df_gt = pd.read_csv(gt_csv, header=None)
    correct_list=[]
    pre=df_pre.values
    gt=df_gt.values

    #画像の名前リスト
    name_list=pre[:,0]

    #name_label部分を削除&型を指定
    #推定
    pre=np.delete(pre,0,1)
    pre=pre.astype(np.float64)
    #正解
    gt=np.delete(gt,0,1)
    gt=gt.astype(np.float64)

    for x,y,z in zip(pre,gt,name_list):
        if af.Class(y,x) == 0:
            correct_list.append(z)
        else:
            pass
    
    df_ci = pd.DataFrame(correct_list)

    #モデルの名前を入力
    print("model name:")
    name_header = input()
    name_output=name_header+'('+str(datetime.datetime.now())+')'+".csv"
    df_ci.to_csv(name_output,index=False, header=False)

if __name__ == "__main__":
    print("one_analysis(1), correct_image(2) or five_analysis(3):")
    choice=input()

    if choice == '1':
        print("result csv:")
        pre_csv=input()
        print("ground truth csv:")
        gt_csv=input()
        print("input name:")
        analysis_name=input()
        one_analisis(pre_csv,gt_csv,analysis_name)
    elif choice == '2':
        print("result csv:")
        pre_csv=input()
        print("ground truth csv:")
        gt_csv=input()
        correct_image(pre_csv,gt_csv)
    elif choice == '3':
        five_analysis()
    else:
        print("Choice Error!")