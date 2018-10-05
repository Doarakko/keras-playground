# -*- coding: utf-8 -*-
import glob
import re
import json
import shutil
from PIL import Image
import numpy as np

import train


# 画像サイズ
IMAGE_SIZE = 224
# チャネル数
CHANNEL_SIZE = 3
# 出力数
N_OUT = len(glob.glob('image/*'))


# 学習済みのモデルをロードする関数
def load_model(in_shape):
    # モデルを構築
    model = train.build_model(in_shape)
    # 学習済みのモデルをロード
    model.load_weights("./log/model.hdf5")
    return model


# ラベルの辞書を取得する関数
def get_label_dic():
    # ラベルの辞書をロード
    with open("./data/label_dic.json", "r") as f:
        label_json = json.load(f)
        label_dic = json.loads(label_json)
    return label_dic


# 画像を数値データに変換する関数
def convert_image(img_path):
    try:
        # 画像の名前を取得
        image_name = re.search(r'image_predict/(.+)', img_path)
        image_name = image_name.group(1)
        # 白黒画像
        if CHANNEL_SIZE == 1:
            img = Image.open(img_path).convert('L')
        # カラー画像
        else:
            img = Image.open(img_path).convert('RGB')
        # 画像サイズを変換
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        # 画像データをnumpy配列に変換
        x = np.asarray(img, dtype=np.float32)
        # 正規化
        x /= 255
        return x
    except Exception as e:
        shutil.move(img_path, "noise")
        x = []
        return x


# 推論する関数
def predict(model, x_list, label_dic, img_path_list):
    y_list = model.predict(x_list)
    for i, y in enumerate(y_list):
        # 画像の名前を取得
        image_name = re.search(r'image_predict/(.+)', img_path_list[i])
        image_name = image_name.group(1)
        # 辞書ビューオブジェクトでキーを取得
        keys_dic_view = label_dic.keys()
        # 辞書ビューオブジェクトをリストに変換
        val = list(keys_dic_view)[y.argmax()]

        print('image: {0}\tpredicted label: {1}\tvalue: {2}'.format(image_name.ljust(30, ' '), y.argmax(), val))


if __name__ == '__main__':
    # ラベルの辞書を取得
    label_dic = get_label_dic()

    # データセットをロード
    train_image, train_label, test_image, test_label = train.load_dataset()

    # 入力サイズ
    in_shape = train_image.shape[1:]

    # 学習済みのモデルをロード
    model = load_model(in_shape)

    # 画像のパスのリストを取得
    img_path_list = glob.glob('image_predict/*.jpg')
    x_list = []
    for img_path in img_path_list:
        # 画像を数値データに変換
        x = convert_image(img_path)
        if x == []:
            continue
        x_list.append(x)
    x_list = np.array(x_list)
    # 推論
    predict(model, x_list, label_dic, img_path_list)
