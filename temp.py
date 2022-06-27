#!/usr/bin/env python
# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error
import requests
import base64
import json
from playsound import playsound
from aip import AipSpeech
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 调用Baidu API鉴权
APP_ID = '26543514'
API_KEY = 'EXfctgN6vHUm17AMAAmmB27h'
SECRET_KEY = 'YZy6b6xiTylOSZvv1WQuC0SEmHEdoZlI'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


def show_score(model):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    print("训练集平均绝对误差", mean_absolute_error(Y_train, train_preds))
    print("测试集平均绝对误差", mean_absolute_error(Y_test, test_preds))
    print("训练集均方误差", mean_squared_log_error(Y_train, train_preds))
    print("测试集均方误差", mean_squared_log_error(Y_test, test_preds))
    print("训练集均方根误差", np.sqrt(mean_squared_log_error(Y_train, train_preds)))
    print("测试集均方根误差", np.sqrt(mean_squared_log_error(Y_test, test_preds)))
    print("训练集R2分数", r2_score(Y_train, train_preds))
    print("测试集R2分数", r2_score(Y_test, test_preds))
    return 0


# 手写文字识别
def readHandWriting(input):
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=68ZuEfFeO7U7dN1ezNoLhPSf&client_secret=zTkrWZEl4jQ9HsMSFCiyWCsqIKtdqVnF'
    response = requests.get(host)
    if (0 == response):
        return
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting"
    f = open(input, 'rb')
    img = base64.b64encode(f.read())
    params = {"image": img}
    access_token = response.json().get("access_token")
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print(response.json())
    img_output = np.zeros([8, 2])
    hw_res = np.arange(0, 8, 1)
    # 类型转换
    hw_res = hw_res.astype(float)
    for _ in range(8):
        tmp = response.json().get("words_result")[_].get("words")
        if tmp == 'M':
            tmp = '2'
        elif tmp == 'F':
            tmp = '0'
        elif tmp == 'I':
            tmp = '1'
        img_output[_, 0] = response.json().get("words_result")[_].get("location").get("left")
        img_output[_, 1] = tmp
    print(img_output)
    # 冒泡，小到大
    for i in range(len(hw_res)):
        for j in range(len(hw_res) - i - 1):
            if img_output[j, 0] > img_output[j + 1, 0]:
                img_output[j, 0], img_output[j + 1, 0] = img_output[j + 1, 0], img_output[j, 0]
                img_output[j, 1], img_output[j + 1, 1] = img_output[j + 1, 1], img_output[j, 1]
    for _ in range(len(hw_res)):
        hw_res[_] = img_output[_, 1]
    print(hw_res)
    return hw_res


# 语音合成
def speechSync(input):
    result = client.synthesis('预测出的鲍鱼年龄约为' + input + '岁', 'zh', 1, {'vol': 5, })
    if not isinstance(result, dict):
        with open('data/audio.mp3', 'wb') as f:
            f.write(result)


if __name__ == '__main__':
    data = pd.read_csv("data/data_bak.txt")
    l_encoder = LabelEncoder()
    data["Sex"] = l_encoder.fit_transform(data["Sex"])
    X = data.drop("Rings", axis=1)
    Y = data["Rings"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14)
    # 使用搜索后的结果建立预测模型
    ideal_model = RandomForestRegressor(max_depth=None, max_features=0.5, min_samples_leaf=2, min_samples_split=10,
                                        n_estimators=200)
    ideal_model.fit(X_train, Y_train)

    # 模型参数打印
    # show_score(ideal_model)

    # 测试用例
    # handwrite = [2, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15]
    # a = np.array(handwrite)
    # a = a.reshape(1, -1)
    # test_pred = ideal_model.predict(X_test)
    # print(test_pred[:20])
    # print(Y_test[:20])
    # test_pred = ideal_model.predict(a)
    # print(test_pred)
    # 手写识别
    handwrite = readHandWriting('hw1.jpg')
    handwrite = np.array(handwrite)
    handwrite = handwrite.reshape(1, -1)
    print("手写数字识别结果为：", handwrite)
    # 进行预测
    res = ideal_model.predict(handwrite)
    print("预测鲍鱼年龄约为：" + str(np.round(res)) + '(' + str(res) + ')')

    # 语音合成
    speechSync(str(np.round(res)))
    # 语音播放
    playsound('audio.mp3')


def save(model):
    joblib.dump(model, 'pkl/detector.pkl')
