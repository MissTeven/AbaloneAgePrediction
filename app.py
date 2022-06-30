import logging

import flask
import numpy as np
from flask import request, jsonify, json

from detector import Detector

server = flask.Flask(__name__)
detector = Detector()


@server.route('/', methods=['get'])
def testGet():
    return {"code": 200, "message": "ok"}


@server.route('/test/post', methods=['post'])
def testPost():
    print("test")
    return "test"


@server.route('/detect', methods=['post'])
def detect():
    print(request.headers)
    # 将bytes类型转换为json数据
    data = str(request.data, 'UTF-8')
    print(data)
    temp = data.split("&")
    dict = {}
    for item in temp:
        key, value = item.split("=")
        dict[key] = value
    sex = dict.get('sex')
    if sex == "M":
        sex = 2
    elif sex == "F":
        sex = 0
    else:
        sex = 1
    length = dict.get('length')
    diameter = dict.get('diameter')
    height = dict.get('height')
    whole_weight = dict.get('whole_weight')
    shucked_weight = dict.get('shucked_weight')
    viscera_weight = dict.get('viscera_weight')
    shell_weight = dict.get('shell_weight')
    data = [sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]
    print(data)
    data = np.asarray(data).astype(float)
    handwrite = np.array(data)
    data = handwrite.reshape(1, -1)
    res = np.round(detector.detect(data)).sum()
    print("predict result->", res)
    return jsonify({"code": 200, "message": "ok", "data": res})
