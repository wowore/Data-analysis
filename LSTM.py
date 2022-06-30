import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
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


# 下面是如滑窗大小为2，那么序列1，2，3，滑窗[1,2],3就把3作为它的一个标签
def create_dataset(x, y, seq_len=10):  # 步长设置为10，即隔10天取一组数据
    features, targets = [], []
    for i in range(0, len(x) - seq_len, 1):  # 滑动窗口，1表示每一步滑一次
        data = x.iloc[i:i + seq_len]  # 序列数据,iloc[:,:]函数:','前是行，后是列，':'遵循左闭右开原则
        label = y.iloc[i + seq_len]  # 标签数据,每一组10个数据中的最后一个作为标签
        # 保存到features,targets中
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)  # 返回的必须是array类型


class LSTMprocessing:
    def __init__(self):
        self.path = "数据文件\\predict-data.csv"
        self.dataset = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date',
                                   parse_dates=['date'])

        self.yy = ['total_deaths', 'new_deaths', 'total_cases', 'new_cases',
                   'total_cases_per_million', 'new_cases_per_million',
                   'total_deaths_per_million', 'new_deaths_per_million', ]  # 取出来的数据类型

        self.a = 'total_cases'
        self.name = 'AFG'
        self.weilai = None
        self.pred = None

        pd.set_option('display.unicode.east_asian_width', True)
        self.dataset.dropna(inplace=True)
        # print(type(self.dataset))

        self.scalera = None  # 归一化模型
        self.model = None  # 训练模型LSTM

        # plt.figure(figsize=(8, 8))
        # sns.pointplot(x='total_deaths', y='total_cases', data=self.dataset)
        # plt.show()

    def go(self):
        self.division()
        self.weilai = self.dataset[self.a]
        self.normalization()
        self.FuturePrediction()

    def division(self):  # 选取要分析的数据
        # 分割数据集
        self.dataset = self.dataset.loc[self.dataset['iso_code'] == self.name, self.yy]
        # 数据表格对齐
        pd.set_option('display.unicode.east_asian_width', True)
        # 删除空数据
        self.dataset.dropna(inplace=True)

    def normalization(self):  # 归一化
        for y in self.yy:
            scaler = MinMaxScaler()  # 归一化模型,归一化,反归一化要用的
            # 注意，下面的归一化别写错了，因为要有时间序列，和只有数据的归一化不同，reshape中-1是表示不论多少行，自动统计，1表示一列
            self.dataset[y] = scaler.fit_transform(self.dataset[y].values.reshape(-1, 1))
            if y == self.a:
                self.scalera = scaler
        #     print(y, '\n:', self.dataset[y])
        # print("归一化后:\n", self.dataset)

    def FuturePrediction(self):
        # 分割自变量和因变量
        x = self.dataset.drop(columns=[self.a], axis=1)  # 特征集 取出除了要被预测的特征列之外的所有列
        y = self.dataset[self.a]  # 标签集 取出要预测的特征列
        # print("y=\n", y)

        # 数据分离,注意shuffle一定要设置成False，默认为True，会将数据打乱，但我们需要时间序列，就不能打乱
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=666)
        # print("x_train", x_train.shape)
        # print(x_train)
        # print("y_test", y_test.shape)
        # print(y_test)

        # 构造训练特征数据集
        seq = 2
        train_dataset, train_labels = create_dataset(x_train, y_train, seq_len=seq)  # 为1即可不用取滑动窗口，但又可以使得维度变到3
        # 滑窗数，每个滑窗数据量，每一个数据的特征数(维度)
        # print("train_dataset\n", train_dataset)
        # print("train_labels\n", train_labels)
        # print(train_dataset.shape, train_labels.shape)

        # 构造测试特征数据集
        test_dataset, test_labels = create_dataset(x_test, y_test, seq_len=seq)  # 为1即可不用取滑动窗口，但又可以使得维度变到3
        # 滑窗数，每个滑窗数据量，每一个数据的特征数(维度)
        # print(test_dataset.shape, test_labels.shape)
        # print("test_dataset:\n", test_dataset)

        # 模型搭建
        # LSTM:units：输出维度,return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
        # input_shape有三个维度，但一般第一个维度样本个数不需要去写，填后两个，一个是时间步长(窗口大小)，一个是特征维度,用train_dataset.shape[-2:]即可直接获取
        # return_sequences是需要将每一次训练之后的状态往后传，建立多个模型的时候需要
        model = Sequential([
            layers.LSTM(units=256, input_shape=train_dataset.shape[-2:], return_sequences=True),
            layers.Dropout(0.4),  # 删除部分神经元
            layers.LSTM(units=256, return_sequences=True),
            layers.Dropout(0.3),  # 删除部分神经元
            layers.LSTM(units=128, return_sequences=True),
            layers.LSTM(units=32),
            layers.Dense(1)
        ])
        # 模型编译adam是一个优化器，mse是衡量标准
        model.compile(optimizer='adam', loss='mse')
        # 存储最好的模型
        checkpoint_file = "best_model.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                              moniter='loss',
                                              mode='min',
                                              save_best_only=True,
                                              save_weights_only=True)
        # 开始训练
        history = model.fit(train_dataset, train_labels,
                            epochs=30,
                            validation_data=(test_dataset, test_labels),
                            callbacks=[checkpoint_callback],
                            )

        # 拟合结果
        # plt.figure(figsize=(16, 8))
        # plt.plot(history.history['loss'], label='train loss')
        # plt.plot(history.history['val_loss'], label='val loss')
        # plt.legend(loc='best')
        # plt.show()

        # 模型验证
        # print("test_dataset\n")
        # print(test_dataset.shape)
        print("开始预测:\n")
        # 注意，如果需要用其他的数据进行预测的时候，需要走一遍流程，归一化、array化，注意np.array函数是把数组直接变成array类型，所以下面测试
        # 可以用np.array([test_dataset[0]]),报错是因为下面r2的错
        test_preds = model.predict(test_dataset, verbose=1)
        # print("test_preds.shape\n")
        # print(test_preds.shape)
        # print(test_preds)

        # 计算r2,一般60~70就优秀了，主要看模型
        score = r2_score(test_labels, test_preds)
        # print("r2=", score)

        # 这里还缺少逆归一化，需要将dataset的date索引加到test_labels和test_preds(它们是array类型),把它们变成pandas类型才可以
        test_labels = pd.DataFrame(test_labels, index=y_test.index[seq:])
        test_preds = pd.DataFrame(test_preds, index=y_test.index[seq:])
        # 进行归一化
        # print("test_labels:\n", test_labels, type(test_labels))
        test_labels = self.scalera.inverse_transform(test_labels)
        # print("test_labels:\n", test_labels, type(test_labels))
        test_preds = self.scalera.inverse_transform(test_preds)
        # print("test_preds:\n", test_preds, type(test_preds))
        # 归一化后会失去下标，变成其他类型，为了使得最后X轴能是日期，就需要再次将他变成pandas类型
        test_labels = pd.DataFrame(test_labels, index=y_test.index[seq:])
        test_preds = pd.DataFrame(test_preds, index=y_test.index[seq:])

        # 绘制，预测与真值的结果
        # plt.figure(figsize=(8, 8))
        # plt.plot(test_labels, label="True value")
        # plt.plot(test_preds, label="Pred value")
        # plt.show()

        self.model = model

    def Predict(self):
        scalera = None
        dataset = pd.read_csv(self.path, encoding="utf-8_sig", index_col='date', parse_dates=['date'])
        for y1 in self.yy:
            scaler = MinMaxScaler()  # 归一化模型,归一化,反归一化要用的
            # 注意，下面的归一化别写错了，因为要有时间序列，和只有数据的归一化不同，reshape中-1是表示不论多少行，自动统计，1表示一列
            dataset[y1] = scaler.fit_transform(dataset[y1].values.reshape(-1, 1))
            if y1 == self.a:
                scalera = scaler

        x = dataset.drop(columns=[self.a], axis=1)
        y = dataset[self.a]

        seq = 2
        test, label = create_dataset(x, y, seq_len=seq)

        print("test=\n", test)
        print("weilai=\n", self.weilai)

        pred = self.model.predict(test, verbose=1)
        # 为了把因为滑窗而去掉的seq个值放回去
        i = 0
        while i < seq:
            pred = np.insert(pred, 0, [y.values[i]], 0)
            i += 1
        # 反归一化
        pred = scalera.inverse_transform(pred)
        pred = pd.DataFrame(pred, index=dataset.index)
        print("pred=\n", pred)

        pred.iloc[0, 0] = self.weilai.iloc[-1]  # 变第一个值

        # 绘制，预测与真值的结果
        plt.figure(figsize=(6, 6))
        plt.plot(self.weilai, label="True value")
        plt.plot(pred, label="Pred value")
        plt.savefig(f'结果文件\\{"LSTM"}\\predict-data1{self.name}.PNG')
        # plt.show()

        self.pred = pred


if __name__ == '__main__':
    data = LSTMprocessing()
    # data.yy = ['total_cases', 'total_deaths_per_million', 'total_deaths', 'total_cases_per_million']
    data.yy = ['total_cases', 'total_deaths']
    data.name = 'CHN'
    data.go()
    data.Predict()
