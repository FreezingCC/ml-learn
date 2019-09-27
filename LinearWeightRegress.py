from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class LinearWeightRegression(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        result = []
        for example in X:
            prediction = self._predict_single(example)
            result.append(prediction)
        return result

    def _predict_single(self, x):
        """
        1. 算距离，距离转化成向量
        2. 距离转化成权重
        3. 权重， X，y输入LR
        4. 预测
        :param x:
        :return:
        """
        distance = self.X - x
        weight = np.exp(-np.sum(distance.T ** 2)/(self.k**2))
        model = LinearRegression()
        model.fit(self.X, self.y, sample_weight=weight)
        return model.predict((x, ))[0]


if __name__ == '__main__':
    model = LinearWeightRegression(2)
    data = pd.read_csv('dataset/height_train.csv')
    train, test = data[:int(data.shape[0]/3*2)], data[int(data.shape[0]/3*2):]
    X, y = train.loc[:, ['father_height', 'mother_height']], train.child_height.values
    test_X, test_y = test.loc[:, ['father_height', 'mother_height']], test.child_height.values
    model.fit(X, y)
    s = np.array([model.predict(test_X.values), test_y]).T
    with open('predict.csv', 'a') as f:
        for i in s:
            f.write(str(i[0]) + '\t' + str(i[1]) + '\n')