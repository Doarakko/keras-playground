# -*- coding: utf-8 -*-
import glob
import json
import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import keras_model

# バッチサイズ
BATCH_SIZE = 10
# エポック数
MAX_EPOCH = 10
# 出力数
N_OUT = len(glob.glob('image/*'))


# データセットをロードする関数
def load_dataset():
    image_data = np.load("./data/image_data.npy")
    label_data = np.load("./data/label_data.npy")
    # 学習データとテストデータに分割
    train_image, test_image, train_label, test_label = train_test_split(image_data, label_data, train_size=0.8, shuffle=True)
    # ラベルをone-hot-label形式に変換
    train_label = np_utils.to_categorical(train_label, N_OUT)
    test_label = np_utils.to_categorical(test_label, N_OUT)

    print("train_data: {0}\ttest_data: {1}".format(len(train_image), len(test_image)))
    return train_image, train_label, test_image, test_label


# モデルを構築する関数
def build_model(in_shape):
    N_OUT = len(glob.glob('image/*'))
    model = keras_model.my_model(in_shape, N_OUT)
    return model


# 学習する関数
def train_model(model, x, y):
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=MAX_EPOCH, shuffle=True)
    # モデルを保存するパス
    save_model_path = "./log/model.hdf5"
    # モデルを保存
    model.save_weights(save_model_path)
    return model


# 評価する関数
def evaluate_model(model, x, y):

    print("test loss: {:.04f}\ttest accuracy: {:.04f}".format(score[0], score[1]))

if __name__ == '__main__':
    # データセットをロード
    train_image, train_label, test_image, test_label = load_dataset()

    # 入力サイズ
    in_shape = train_image.shape[1:]

    # モデルを構築
    model = build_model(in_shape)
    # モデルを可視化
    plot_model(model, to_file='./log/model.png')
    # 学習
    model_train = train_model(model, train_image, train_label)
    # 評価
    evaluate_model(model_train, test_image, test_label)
