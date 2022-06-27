import joblib


class Detector:
    def __init__(self):
        self.mode = joblib.load('pkl/detector.pkl')

    def detect(self, data):
        return self.mode.predict(data)
