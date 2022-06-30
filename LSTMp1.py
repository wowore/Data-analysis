import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras import layers

import tensorflow as tf
import warnings

'''要求：当前设置的滑动窗口为1，那么取出来的数据第一个日期会被丢弃,所以存储未来数据时需要多存储之前的一些，最后输出的时候'''
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


def predict_next(model, sample, seq_len, epoch=20):
    temp1 = list(sample[:, 0])
    print("进入时:temp1=", temp1)
    for i in range(epoch):
        print("转换前sample=", sample)
        sample = sample.reshape(1, seq_len, 1)
        print("转换后sample=", sample)
        pred = model.predict(sample)
        print("pred=", pred)
        value = pred.tolist()[0][0]  # 转换
        print("value=", value)
        temp1.append(value)
        print("temp1=", temp1)
        sample = np.array(temp1[i + 1:i + seq_len + 1])
        print("添加后sample=", sample, "坐标:", i + 1, ':', seq_len + 1)

    # 取出最后的那部分数据，也就是未来的数据，用seq_len或者epoch来分割都可以
    temp1 = np.array(temp1[-epoch - 1:])
    temp1 = temp1.reshape(1, epoch + 1, 1)
    print("新temp1=", temp1)
    return temp1


def create_dataset(dataset, look_back):  # 设置滑窗
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    # print("dataset.shape", dataset.shape, len(dataset), dataset[len(dataset) - 1])
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        # print("i=", i, 'i + look_back=', i + look_back, ':', a, "Y=", dataset[i + look_back])
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    # print("长度\n", len(dataX), len(dataY))
    return np.array(dataX), np.array(dataY)


class LSTMprocessing:
    def __init__(self):
        self.dataset = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date',
                                   parse_dates=['date'])

        self.a = 'total_cases'
        self.name = 'MDG'

        pd.set_option('display.unicode.east_asian_width', True)
        self.dataset.dropna(inplace=True)
        # print(type(self.dataset))

        self.scaler = MinMaxScaler()  # 归一化模型

        self.preds = 0

        # plt.figure(figsize=(6, 6))
        # sns.pointplot(x='total_deaths', y='total_cases', data=self.dataset)
        # plt.show()

    def go(self):
        self.division()
        self.normalization()
        self.FuturePrediction()

    def division(self):  # 选取要分析的数据
        # 分割数据集
        self.dataset = self.dataset.loc[self.dataset['iso_code'] == self.name, [self.a]]
        # print(self.dataset)
        # 数据表格对齐
        pd.set_option('display.unicode.east_asian_width', True)
        # 删除空数据
        self.dataset.dropna(inplace=True)

    def normalization(self):  # 归一化
        self.dataset[self.a] = self.scaler.fit_transform(self.dataset[self.a].values.reshape(-1, 1))
        # print("归一化:\n", self.dataset[self.a])

    def FuturePrediction(self):
        dataset = self.dataset.values

        # 划分
        train_size = int(len(dataset) * 0.8)
        trainlist = dataset[:train_size]
        testlist = dataset[train_size:]
        test_labels = self.dataset[train_size:]

        look_back = 10
        trainX, trainY = create_dataset(trainlist, look_back)
        testX, testY = create_dataset(testlist, look_back)
        # print("xtrain:\n", trainX)
        # print("xtest:\n", testX)
        # print("ytrain:\n", trainY)
        # print("ytest:\n", testY.shape)

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        # print("xtrain转换后:\n", trainX)
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        model = Sequential([
            # layers.LSTM(units=128, input_shape=(None, 1)),
            layers.LSTM(units=100, input_shape=(look_back, 1)),
            layers.Dense(1)
        ])
        # 模型编译adam是一个优化器，mse是衡量标准
        model.compile(optimizer='adam', loss='mse')  # mae,mse

        # parameters = {'batch_size': [16, 20], 'epochs': [8, 10], 'optimizer': ['adam', 'Adadelta']}
        #         # grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=2)
        #         # grid_search.fit(trainX, trainY)
        #         # print(grid_search.best_params_)  # 这里是想做自动调参，但是报错说缺少评分函数，暂时先手动调参

        # 存储最好的模型
        checkpoint_file = "best_model.hdfp1"
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                              moniter='loss',
                                              mode='min',
                                              save_best_only=True,
                                              save_weights_only=True)
        # 开始训练
        history = model.fit(trainX, trainY,
                            epochs=50,  # 训练次数
                            batch_size=200,  # 超参数之一，一般50-400左右,批数据量
                            validation_data=(testX, testY),
                            callbacks=[checkpoint_callback],
                            )

        # 拟合结果，两个参数，越小越好
        plt.figure(figsize=(6, 6))
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend(loc='best')
        plt.show()

        testPredict = model.predict(testX)

        # 反归一化
        # print("testPredict反归一化前\n", testPredict)
        testPredict = self.scaler.inverse_transform(testPredict)
        # print("testPredict反归一化后\n", testPredict)
        # print("testY反归一化前\n", testY)
        testY = self.scaler.inverse_transform(testY)
        # print("testY反归一化后\n", testY)

        # 回归效果
        testPredict = pd.DataFrame(testPredict, index=test_labels.index[look_back:])
        testY = pd.DataFrame(testY, index=test_labels.index[look_back:])

        plt.plot(testY, label='真实数据')
        plt.plot(testPredict, label='预测数据')  # 测试集训练
        plt.legend(loc='best')
        plt.show()

        # 预测未来day天
        day = 7
        trueY = self.scaler.inverse_transform(self.dataset)  # 获取真实数据，因为之前的self.dataset被归一化了，需要反归一化
        trueY = pd.DataFrame(trueY, index=self.dataset.index)

        true = testX[-1]  # 利用最后一个滑窗的数据往后进行预测day天数的数据
        print("true:\n", true)
        print("原本的数据\n", testX)
        preds = predict_next(model, true, look_back, day)[0]
        # print("新preds=", preds)

        # 生成未来日期,作为下标使用
        # start_date = "2022-3-12"
        # end_date = "2022-3-19"
        # date = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        date = pd.date_range("2022-03-11", periods=day + 1, freq="D")
        print(date)
        preds = self.scaler.inverse_transform(preds)
        preds = pd.DataFrame(preds, index=date)

        preds.iloc[0, 0] = trueY.iloc[-1]  # 变第一个值
        print("最终获得=", preds)

        # 预测结果
        plt.figure(figsize=(6, 6))
        plt.plot(preds, color='blue', label='预测数据')
        plt.plot(testY, color='red', label='真实数据')  # 测试集数据
        plt.plot(trueY, color='yellow', label='真实数据')  # 原始数据
        plt.legend(loc='best')
        plt.savefig(f'结果文件\\{"LSTMp1"}\\predict-data1{self.name}.PNG')
        plt.show()
        preds.to_csv(f'结果文件\\{"LSTMp1"}\\predict-data1{self.name}.csv', index=False, encoding="utf-8_sig")

        self.preds = preds


if __name__ == '__main__':
    data = LSTMprocessing()
    data.name = 'JPN'
    data.a = 'total_cases'
    # data.a = 'total_deaths'
    data.go()
# JPN:100 50 200
# CHN: 8 500 50
# AFG 100 50 50
# DEU 100 50 400
# USA 100 50 400
