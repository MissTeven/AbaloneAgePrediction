import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Trainer:
    def __init__(self):
        self.dataPath = "data.txt"

    def train(self):
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
        self.save(ideal_model)

    def save(self, model):
        joblib.dump(model, 'pkl/detector.pkl')
