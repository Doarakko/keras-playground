#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, re, json, random
from PIL import Image
import numpy as np
from keras.utils import np_utils

#ラベルを作成する関数
def make_label_list():
    #ディレクトリのパスを取得
    dir_path_list = glob.glob('image/*')
    #辞書を準備
    label_dic = {}
    for i, dir_path in enumerate(dir_path_list):
        key = re.search(r'image/(.+)', dir_path)
        key = key.group(1)
        label_dic[key] = i
    #辞書をjsonで保存
    with open("./data/label_dic.json", "w") as f: 
        label_json = json.dumps(label_dic)
        json.dump(label_json, f)
    return label_dic

#画像を数値データに変換する関数
def convert_image(img_path, img_size, ch_size):
    #画像の名前を取得
    image_name = re.search(r'image/(.+)', img_path)
    image_name = image_name.group(1)
    #白黒画像
    if ch_size == 1:
        img = Image.open(img_path).convert('L') 
    #カラー画像
    else:
        img = Image.open(img_path).convert('RGB') 
    #画像サイズを変換
    img = img.resize((img_size, img_size)) 
    #画像データをnumpy配列に変換
    x = np.asarray(img, dtype=np.float32)
    #正規化
    x /= 255
    return x

#ラベルデータを取得する関数
def get_label_data(img_path, label_dic):
    #画像のディレクトリのパスを取得
    key = re.search(r'image/(.+)/.+', img_path)
    key = key.group(1)
    #辞書からラベルを取得
    t = label_dic[key]
    #ラベルをnumpy配列に変換
    t = np.asarray(t, dtype=np.int32)
    return t

#データセットを作成する関数
def make_dataset(label_dic, img_size, ch_size):
    #デバック
    #各画像の枚数を出力
    dir_path_list = glob.glob('image/*')
    for dir_path in dir_path_list:
        img_path_list = glob.glob(dir_path+'/*.jpg')
        dir_name = re.search(r'image/(.+)', dir_path)
        dir_name = dir_name.group(1)
        print("{0}: {1}".format(dir_name, len(img_path_list)))
    print("")

    #画像のパスのリストを取得
    img_path_list = glob.glob('image/*/*.jpg')
    #画像をシャッフル
    random.shuffle(img_path_list)
    #画像データを入れるリストを準備
    image_data = []
    #ラベルデータを入れるリストを準備
    label_data = []
    for img_path in img_path_list:
        #画像を数値データに変換
        x = convert_image(img_path, img_size, ch_size)
        #ラベルデータを取得
        t = get_label_data(img_path, label_dic)
        #リストに追加
        image_data.append(x)
        label_data.append(t)
    #デバック
    print("total: {0}".format(len(img_path_list)))
    return image_data, label_data

if __name__ == '__main__':
    #画像サイズ
    img_size = 100
    #チャネル数
    ch_size = 3
    #ラベルを作成
    label_dic = make_label_list()
    #データセットを作成
    image_data, label_data = make_dataset(label_dic, img_size, ch_size)

    #画像データを保存するパス
    save_image_path = "./data/image_data.npy"
    #ラベルデータを保存するパス
    save_label_path = "./data/label_data.npy"
    
    #画像データをファイルに保存
    np.save(save_image_path, image_data)
    #ラベルデータをファイルに保存
    np.save(save_label_path, label_data)