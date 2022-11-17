import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection

sports = pd.read_csv(r'Run or Walk.csv')
predictors = sports.columns[4:]
X = sports[:][predictors]

y = sports.activity

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, 
                y, test_size = 0.25, random_state = 1234)

# 训练模型
sklearn_logistic = linear_model.LogisticRegression().fit(X_train, y_train)
# 返回模型的各个参数
print('logistic模型的β系数：\n',sklearn_logistic.intercept_, sklearn_logistic.coef_)

# 模型预测
sklearn_predict = sklearn_logistic.predict(X_test)
res = pd.Series(sklearn_predict).value_counts()

print('预测结果：\n', res)

def func1():
    
    return
    
    
    

