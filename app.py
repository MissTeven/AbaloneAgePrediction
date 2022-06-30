import flask
import numpy as np
from flask import request, app

from detector import Detector

server = flask.Flask(__name__)
detector = Detector()


@server.route('/test/get', methods=['get'])
def testGet():
    return {"code": 200, "message": "ok"}


@server.route('/test/post', methods=['post'])
def testPost():
    return "Hello world!"


@server.route('/detect', methods=['post'])
def detect():
    # Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell weight,Rings
    sex = request.form.get('sex')
    if sex == "M":
        sex = 2
    elif sex == "F":
        sex = 0
    else:
        sex = 1

    length = request.form.get('length')
    diameter = request.form.get('diameter')
    height = request.form.get('height')
    whole_weight = request.form.get('whole_weight')
    shucked_weight = request.form.get('shucked_weight')
    viscera_weight = request.form.get('viscera_weight')
    shell_weight = request.form.get('shell_weight')
    rings = request.form.get('rings')

    data = [sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings]
    res = detector.detect(np.asarray(data))
    return {"code": 200, "message": "ok", data: res}
