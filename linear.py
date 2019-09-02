import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression


def LinearRegression_(X, y, alpha, n_rounds):
    beta = np.zeros(shape=[X.shape[1] + 1])
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    for i in range(n_rounds):  #  训练轮数
        beta = beta - alpha * 2 * 1/X.shape[0] * (beta.dot(X.T) - y).T.dot(X)
    return beta


if __name__ == '__main__':
    data = pd.read_csv('dataset/height_train.csv')
    train, test = data[:int(data.shape[0]/3*2)], data[int(data.shape[0]/3*2):]
    X, y = train.loc[:, ['father_height', 'mother_height']], train.child_height.values
    # w = LinearRegression_(X, y, 0.05, 1000000)
    # print(w)
    # [0.80539654 0.25986578 0.27348456]

    # l = LinearRegression()
    # l.fit(X, y)
    # print(l.coef_)
    # print(l.intercept_)
    # [0.805396537397223, 0.25986578, 0.27348456]
