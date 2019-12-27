# coding:utf-8
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import trange, tqdm
import os
import matplotlib.pyplot as plt
import datetime

# from Tool import gradcam
from Tool import analyzer_function as af
from Tool import analyzer as alz
from exp_func import load_Img, load_AttnImg_CASNet

#コマンドライン引数の設定
def argparser():
  parser = argparse.ArgumentParser()

  #GPU setting
  parser.add_argument("--model_name", "-mn", type=str, help="input model name")
  parser.add_argument("--weight", "-w", type=str, help="input weight name")
  parser.add_argument("--img_csv", "-ic", type=str, help="input csv file")
  parser.add_argument("--gradcam", "-g", type=str, help="True or False")
  parser.add_argument("--header", "-he", type=str, help="input header name")
  parser.add_argument("--result_csv", "-rc", type=str, help="input csv file")
  parser.add_argument("--analysis_csv1", "-ac1", type=str, help="input csv file")
  parser.add_argument("--analysis_csv2", "-ac2", type=str, help="input csv file")
  parser.add_argument("--img_type", "-it", type=str, help="correct or incorrect")
  return parser.parse_args()

def extra_test_img(model_name, weight, img_csv, result_name, analysis_name, gcam, header):
  #model作成
  if(model_name == 'vgg16'):
    from models import vgg16 as vgg16
    model=vgg16.creat_VGG16_model()
  elif(model_name == 'aenet2way'):
    from models import aenet_2way as ae2
    model=ae2.creat_aenet2way_model()
  elif(model_name == 'resnet50'):
    from models import resnet50 as res50
    model=res50.creat_ResNet50_model()

  model.load_weights(weight)

  #make gradcam directory
  if gcam == "True":
    gcam_dir="{header}/gradcam_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
    os.mkdir(gcam_dir)
    hmap_dir="{header}/heatmap_{time:%Y%m%d%H%M%S}".format(header=header.rsplit('/',1)[0], time=datetime.datetime.now())
    os.mkdir(hmap_dir)

  img_path, data_x, data_y = load_Img(data_path=img_csv, test_split=1.0)

  #prediction
  result_list=model.predict(data_x)
  result_list=np.concatenate([img_path,result_list],1)
  df = pd.DataFrame(result_list)
  df.to_csv(result_name,index=False, header=False)

  #Evaluation Section
  alz.one_analisis(result_name, img_csv, analysis_name)

def extra_test_Attnimg_CASNet(model_name, weight, img_csv, result_name, analysis_name, gcam, header):
  #model作成
  if(model_name == 'vgg16'):
    from models import vgg16 as vgg16
    model=vgg16.creat_VGG16_model()
  elif(model_name == 'aenet2way'):
    from models import aenet_2way as ae2
    model=ae2.creat_aenet2way_model()
  elif(model_name == 'resnet50'):
    from models import resnet50 as res50
    model=res50.creat_ResNet50_model()

  model.load_weights(weight)

  img_path, data_x, data_y = load_AttnImg_CASNet(data_path=img_csv, test_split=1.0)

  #prediction
  result_list=model.predict(data_x)
  result_list=np.concatenate([img_path,result_list],1)
  df = pd.DataFrame(result_list)
  df.to_csv(result_name,index=False, header=False)

  #Evaluation Section
  alz.one_analisis(result_name, img_csv, analysis_name)

def make_analysis2(result_csv, analysis_csv1, heatmap_csv, output_csv):
  # 実験結果csvファイルの読み込み
  df_rc = pd.read_csv(result_csv,header=None)
  df_rc_array = df_rc.values

  #感情確率分布を抽出
  rc_data = df_rc_array[:,1:]

  #各画像の分布の分散
  rc_var = np.array([np.var(x) for x in rc_data])
  
  # 実験結果csvファイルの読み込み
  df_hc = pd.read_csv(heatmap_csv,header=None)
  df_hc_array = df_hc.values
  
  # 形状調整
  df_hc_reshape = df_hc_array.reshape((len(df_hc_array)))

  # ヒートマップの画素を取得
  hc_data = np.array([cv2.imread(x) for x in df_hc_reshape])/255.

  hc_var = np.array([np.var(x) for x in hc_data])
  
  #ヘッダ作成
  rc_header = np.array(["result_variance"])
  hc_header = np.array(["heatmap_variance"])

  # フッタ作成
  rc_footer = np.array([np.mean(rc_var)])
  hc_footer = np.array([np.mean(hc_var)])

  # ヘッダ・フッタの結合
  rc_var = np.hstack([rc_header, rc_var, rc_footer])[:, np.newaxis]
  hc_var = np.hstack([hc_header, hc_var, hc_footer])[:, np.newaxis]

  # 分析結果csvファイルの読み込み
  df_ac = pd.read_csv(analysis_csv1, header=None)
  df_ac_array = df_ac.values
  new_ac_array = np.concatenate([df_ac_array, rc_var, hc_var], axis=1)

  # 書き出し
  df_output = pd.DataFrame(new_ac_array)
  df_output.to_csv(output_csv, index=False, header=False)

#シェルスクリプトと組み合わせて実行
def extract_imgs(heatmap_csv, analysis2_csv, img_type):
  assert img_type == 'correct' or img_type == 'incorrect', "-t : Input ONLY correct or incorrect"
  # ヒートマップcsvファイルの読み込み
  df_hc = pd.read_csv(heatmap_csv,header=None)
  df_hc_array = df_hc.values

  # 分析2csvファイルの読み込み
  df_ac2 = pd.read_csv(analysis2_csv,header=None)
  df_ac2_array = df_ac2.values
  # スライス
  df_ac2_array1 = df_ac2_array[1:-1,0]
  df_ac2_array2 = df_ac2_array[1:-1,4]
  # shape調整
  df_ac2_array1 = df_ac2_array1[:, np.newaxis]
  df_ac2_array2 = df_ac2_array2[:, np.newaxis]

  # カラムを結合
  data = np.concatenate([df_hc_array, df_ac2_array1, df_ac2_array2], axis=1)
  
  if img_type == 'correct':
    hm_cor = [x[0] for x in data if x[2] == '1']
    img_cor = [x[1] for x in data if x[2] == '1']
    
    hm_cor=' '.join(hm_cor)
    print(hm_cor)
    img_cor=' '.join(img_cor)
    print(img_cor)  

  elif img_type == 'incorrect':
    hm_inc = [x[0] for x in data if x[2] == '0']
    img_inc = [x[1] for x in data if x[2] == '0']

    hm_inc=' '.join(hm_inc)
    print(hm_inc)
    img_inc=' '.join(img_inc)
    print(img_inc)

def compare_Img_AttnImg(origin_csv, attn_csv, header):
  assert os.path.exists(origin_csv) == True, "-ic : Input ONLY csv name"
  assert os.path.exists(attn_csv) == True, "-rc : Input ONLY csv name"

  # origin_csvファイルの読み込み
  df_oc = pd.read_csv(origin_csv,header=None)
  origin = df_oc.values
  
  # attn_csvファイルの読み込み
  df_ac = pd.read_csv(attn_csv,header=None)
  attn = df_ac.values

  # インデックスの抽出
  index = origin[1:, 0]

  # それぞれのKLのみを抽出
  origin = origin[1:, 1].astype(float)
  attn = attn[1:, 1].astype(float)

  # オリジナルと提案手法のKLの差
  dif = origin-attn

  # 提案手法の方が優れている画像のインデックス
  success_index = index[dif > 0]
  fail_index = index[dif < 0]

  # 名前設定
  success_name = header + "_success.csv"
  fail_name = header + "_fail.csv"

  # 成功した画像リストを出力
  df_output = pd.DataFrame(success_index)
  df_output.to_csv(success_name,index=False, header=False)

  # 失敗した画像リストを出力
  df_output = pd.DataFrame(fail_index)
  df_output.to_csv(fail_name,index=False, header=False)

def main():
  args = argparser()

  #GPU Configuration
  # config = tf.ConfigProto(
  #     gpu_options=tf.GPUOptions(
  #         per_process_gpu_memory_fraction=args.GPU_USERATE,# GPU using rate
  #         visible_device_list=args.GPU_NUMBER, # GPU number
  #         allow_growth=True
  #     )
  # )
  # set_session(tf.Session(config=config))
    
  print("extra_test_Attnimg(-1), extra_test(0), make analysis2(1), extract correct imgs and incorrect imgs(2), compare origin and ours(3):")
  choice = input()

  if choice == '-1':
    extra_test_Attnimg_CASNet(args.model_name, args.weight, args.img_csv, args.result_csv, args.analysis_csv1, args.gradcam, args.header)
  elif choice == '0':
    extra_test_img(args.model_name, args.weight, args.img_csv, args.result_csv, args.analysis_csv1, args.gradcam, args.header) 
  elif choice == '1':
    make_analysis2(args.result_csv, args.analysis_csv1, args.heatmap_csv, args.output_csv)
  elif choice == '2':
    extract_imgs(args.heatmap_csv, args.output_csv, args.img_type)
  elif choice == '3':
    compare_Img_AttnImg(args.analysis_csv1, args.analysis_csv2, args.header)


if __name__=='__main__':
  main()