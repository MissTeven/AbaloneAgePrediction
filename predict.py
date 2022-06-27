# Import the tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV



# import data
data = pd.read_csv("data_bak.txt")
# 检查是否有异常值
print(data.info())
print(data.isna().sum())


# 将第一列性别转为数字
l_encoder = LabelEncoder()
data["Sex"] =  l_encoder.fit_transform(data["Sex"])
print(data.info())
print(data)

# 将数据集拆分为属性 & 标签
X = data.drop("Rings", axis=1)
Y = data["Rings"]
print("属性：")
print(X)
print("标签：")
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

# 创建RF模型
rf = RandomForestRegressor(n_jobs=-1, random_state=14)

# 定义评估函数
def show_score(model):
    train_preds= model.predict(X_train)
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

rf.fit(X_train, Y_train)
show_score(rf)

# 打印模型参数
print("当前随机森林的参数：")
pprint(rf.get_params())

rf_grid = {"n_estimators":[20, 100, 200],
           "max_depth":[None, 1, 2, 5],
           "max_features":[0.5, 1, "auto", "sqrt"],
           "min_samples_split":[ 2, 5, 10],
           "min_samples_leaf":[1, 2, 3, 5]}

# # n_iter： 模型进行多少种组合
# # cv：每组参数的交叉验证次数
# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
#                                                     random_state=14),
#                                                     param_distributions=rf_grid,
#                                                     n_iter=100,
#                                                     cv=5,
#                                                     verbose=True)
# rs_model.fit(X_train, Y_train)
# # 打印搜索出来的最佳模型的分数
# show_score(rs_model)
# # 打印最佳模型的参数
# print(rs_model.best_params_)

def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                  "feature_importances": importances})
    .sort_values("feature_importances", ascending=False) # sort importances from the biggest to the smallest
    .reset_index(drop=True)) # deletes the index number

    #Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20], color="skyblue")
    ax.set_ylabel("Features")
    ax.set_label("Feature importance")
    ax.invert_yaxis()

ideal_model = RandomForestRegressor(max_depth=None, max_features=0.5, min_samples_leaf=2, min_samples_split=10,
                                    n_estimators=200)
ideal_model.fit(X_train, Y_train)
plot_features(X_train.columns, ideal_model.feature_importances_)
plt.show()


