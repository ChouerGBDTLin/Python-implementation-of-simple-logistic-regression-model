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
print(sklearn_logistic.intercept_, sklearn_logistic.coef_)

# 模型预测
sklearn_predict = sklearn_logistic.predict(X_test)
res = pd.Series(sklearn_predict).value_counts()

print('预测结果：\n', res)

## 模型的评估
from sklearn import metrics

# 混淆矩阵
cm = metrics.confusion_matrix(y_test, sklearn_predict, labels = [0, 1])
print('混淆矩阵：\n', cm)


# 混淆矩阵热力图
import seaborn as sns
sns.heatmap(cm, annot=True, fmt = '.2e', cmap = 'GnBu')


# ROC曲线和AUC数值
# y得分为模型预测正例的概率
plt.figure()
y_score = sklearn_logistic.predict_proba(X_test)[:, 1]

fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
#绘制面积图
plt.stackplot(fpr, tpr, colors='steelblue', alpha = 0.5, edgecolor = 'black')
# 添加ROC曲线的轮廓
plt.plot(fpr, tpr, color = 'black', lw = 1)
# 添加对角线
plt.plot([0, 1], [0, 1], color = 'red', linestyle = '--')
# 添加文本信息
plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)

plt.xlabel('1-Specificity')
plt.ylabel('Specificity')

plt.show()


















