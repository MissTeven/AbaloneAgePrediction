import base64
from io import BytesIO
from PIL import Image
import flask
from flask import request
import numpy as np

server = flask.Flask(__name__)


@server.route('/test/get', methods=['get'])
def testGet():
    return "Hello world!"


@server.route('/test/post', methods=['post'])
def testPost():
    return "Hello world!"


@server.route('/recognize', methods=['GET', 'POST'])
def recognize():
    img = request.form.get('image')

    base_s1 = base64.b64decode(img.encode())  # 将str转换为bytes,并进行base64解码，得到bytes类型
    buf = BytesIO()  # 内存中创建一个buf,用于存储图像文件内容
    buf.write(base_s1)  # 将图像文件内容写入到该buf中，该buf相当于一个临时文件
    buf.seek(0)  # 将文件指针放在文件开头
    data = Image.open(buf).convert('RGB')  # 将buf作为文件名，读取该文件，并转换成RGB
    data = np.mat(data)  # 将图像数据转换成array
    return modeToRecognize(data)


def modeToRecognize(data):
    return "Hello World!"
