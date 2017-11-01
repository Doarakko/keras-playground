#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, re, json
from PIL import Image
import numpy as np
import keras_train as kt
from keras.utils import np_utils

#学習済みのモデルをロードする関数
def load_model(in_shape, n_out):
    #モデルを構築
    model = kt.build_model(in_shape, n_out)
    #学習済みのモデルをロード
    model.load_weights("./log/model.hdf5")
    return model

#画像を数値データに変換する関数
def convert_image(img_path, img_size, ch_size):
    #画像の名前を取得
    image_name = re.search(r'image_predict/(.+)', img_path)
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

#推論する関数
def predict(model, x_list, label_dic, img_path_list):
    #推論
    y_list = model.predict(x_list)
    for i, y in enumerate(y_list):
        #画像の名前を取得
        image_name = re.search(r'image_predict/(.+)', img_path_list[i])
        image_name = image_name.group(1)
        #辞書ビューオブジェクトでキーを取得
        keys_dic_view = label_dic.keys()
        #辞書ビューオブジェクトをリストに変換
        val = list(keys_dic_view)[y.argmax()]   
        #デバック
        print('image: {0}\tpredicted label: {1}\tvalue: {2}'.format(image_name.ljust(30,' '), y.argmax(), val))

if __name__ == '__main__':
    #ラベルの辞書を取得
    label_dic = kt.get_label_dic()
    #クラス数
    class_size = len(label_dic)
    #データセットをロード
    train_image, train_label, test_image, test_label = kt.load_dataset(class_size)
    #画像サイズ
    img_size = 100
    #チャネル数
    ch_size = 3
    #入力サイズ
    in_shape = train_image.shape[1:]
    #出力数
    n_out = len(glob.glob('image/*'))
    #学習済みのモデルをロード
    model = load_model(in_shape, n_out)

    #画像のパスのリストを取得
    img_path_list = glob.glob('image_predict/*.jpg')
    x_list = []
    for img_path in img_path_list:
        #画像を数値データに変換
        x = convert_image(img_path, img_size, ch_size)
        x_list.append(x)
    x_list = np.array(x_list)    
    #推論
    predict(model, x_list, label_dic, img_path_list)