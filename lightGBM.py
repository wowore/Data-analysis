import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt


class GBMprocessing:
    def __init__(self):
        self.path = "数据文件\\predict-data.csv"
        self.df = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date',
                              parse_dates=['date'])

        self.yy = ['total_deaths', 'new_deaths', 'total_cases', 'new_cases',
                   'total_cases_per_million', 'new_cases_per_million',
                   'total_deaths_per_million', 'new_deaths_per_million', ]  # 取出来的数据类型
        self.a = 'total_cases'
        self.name = 'AFG'
        self.model = None  # 训练模型GBM
        self.pred = None

    def go(self):
        self.division()
        self.GBMPrediction()

    def division(self):  # 选取要分析的数据
        # 分割数据集
        self.df = self.df.loc[self.df['iso_code'] == self.name, self.yy]
        # 数据表格对齐
        pd.set_option('display.unicode.east_asian_width', True)
        # 删除空数据
        self.df.dropna(inplace=True)

    def GBMPrediction(self):
        # 分割自变量和因变量
        self.yy.remove(self.a)
        x = self.df.loc[:, self.yy]
        y = self.df[self.a]

        # x = x.values
        # y = y.values
        # 分割训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
        # 构建模型,调整参数非常重要
        model = LGBMRegressor(boosting_type='gbdt',  # 设置提升类型
                              metric='rmse',  # 评估模型
                              objective='regression',  # 设置目标函数
                              max_depth=5,  # 每个基学习器的最大深度
                              feature_fraction=0.8,
                              bagging_freq=5,
                              learning_rate=0.1,
                              n_estimators=2500,  # 基学习器的训练数量.
                              max_bin=255,
                              subsample_for_bin=5000,
                              min_split_gain=0,  # 树的叶子节点上进行进一步划分所需的最小损失减少.
                              min_child_weight=0,  # 就是这里导致最后值变平，越小越好
                              min_child_samples=0,  # 叶节点样本的最少数量(默认20) 就是这里导致最后值变平，越小后面就越不平
                              subsample=1,  # 训练时采样一定比例的数据
                              subsample_freq=1,
                              colsample_bytree=1,
                              num_leaves=32,  # 每个基学习器的最大叶子节点. 大会更准,但可能过拟合
                              max_drop=50,
                              lambda_l2=0.1,  # L2正则化权重项,增加此值将使模型更加保守。
                              )  # 回归器

        # 开始训练
        model.fit(x_train, y_train)
        # 开始预测
        y_pred = model.predict(x_test)
        print('x_test:\n', x_test)

        y_pred = pd.DataFrame(y_pred, index=y_test.index)

        # 重排
        y_test = y_test.sort_index()
        y_pred = y_pred.sort_index()

        print('y_pred:\n', y_pred)
        print('y_test:\n', y_test)

        # 测试精度
        r2 = r2_score(y_test, y_pred)
        print("精度=", r2)

        # 绘制，预测与真值的结果
        # plt.figure(figsize=(6, 6))
        # plt.plot(y_test, label="True value")
        # plt.plot(y_pred, label="Pred value")
        # plt.show()

        self.model = model

    def Predict(self):
        dataset = pd.read_csv(self.path, encoding="utf-8_sig", index_col='date', parse_dates=['date'])
        # dataset = self.df
        x = dataset.loc[:, self.yy]
        # x = x[730:]
        print("x=\n", x)
        y = self.df[self.a]
        # y = y[730:]
        print("y=\n", y)

        # x = x.sample(frac=1)# 乱序

        pred = self.model.predict(x)
        pred = pd.DataFrame(pred, index=dataset.index)
        # pred = pd.DataFrame(pred, index=y.index)
        pred = pred.sort_index()

        pred.iloc[0, 0] = y.iloc[-1]  # 变第一个值
        print("pred=\n", pred)

        # 绘制，预测与真值的结果
        plt.figure(figsize=(6, 6))
        plt.plot(y, label="True value")
        plt.plot(pred, label="Pred value")
        plt.savefig(f'结果文件\\{"GBM"}\\predict-data1{self.name}.PNG')
        # plt.show()

        self.pred = pred


if __name__ == '__main__':
    data = GBMprocessing()
    # data.yy = ['total_cases', 'total_deaths_per_million', 'total_deaths', 'total_cases_per_million']
    data.yy = ['total_cases', 'total_deaths']
    data.name = 'AFG'
    data.go()
    data.Predict()
