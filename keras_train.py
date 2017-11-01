#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

#ラベルの辞書を取得する関数
def get_label_dic():
    #ラベルの辞書をロード
    with open("./data/label_dic.json", "r") as f: 
        label_json = json.load(f)
        label_dic = json.loads(label_json)
    return label_dic

#データセットをロードする関数
def load_dataset(class_size):
    image_data = np.load("./data/image_data.npy")
    label_data = np.load("./data/label_data.npy")
    #学習データとテストデータに分割
    train_image, test_image, train_label, test_label = train_test_split(image_data, label_data, train_size=0.8)
    #ラベルをone-hot-label形式に変換
    train_label = np_utils.to_categorical(train_label, class_size)
    test_label = np_utils.to_categorical(test_label, class_size)
    #デバック
    print("train_data: {0}\ttest_data: {1}\n".format(len(train_image), len(test_image)))
    return train_image, train_label, test_image, test_label

#モデルを構築する関数
def build_model(in_shape, n_out):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_out))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

#学習する関数
def train_model(model, x, y, batch_size, max_epoch):
    model.fit(x, y, batch_size=batch_size, epochs=max_epoch, shuffle=True)
    #モデルを保存するパス
    save_model_path = "./log/model.hdf5"
    #モデルを保存
    model.save_weights(save_model_path)
    return model

#評価する関数
def evaluate_model(model, x, y, batch_size):
    score = model.evaluate(x, y, batch_size, verbose=0)
    #デバック
    print("test loss: {:.04f}\ttest accuracy: {:.04f}".format(score[0], score[1]))

if __name__ == '__main__':
    #クラス数を取得
    class_size = len(get_label_dic())
    #データセットをロード
    train_image, train_label, test_image, test_label = load_dataset(class_size)
    #バッチサイズ
    batch_size = 10
    #エポック数
    max_epoch = 5
    #チャネル数
    #ch_size = 3
    #入力サイズ
    in_shape = train_image.shape[1:]
    #出力数
    n_out = len(glob.glob('image/*'))
    
    #モデルを構築
    model = build_model(in_shape, n_out)
    #モデルを可視化
    plot_model(model, to_file='./log/model.png')
    #学習
    model_train = train_model(model, train_image, train_label, batch_size, max_epoch)
    #評価
    evaluate_model(model_train, test_image, test_label, batch_size)