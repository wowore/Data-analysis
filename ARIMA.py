import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from matplotlib.pylab import style
from tqdm.contrib import itertools
import warnings

warnings.filterwarnings('ignore')

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''pandas取属性：.columns取值：.values'''


class ARIMAprocessing:
    def __init__(self):
        self.df = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date', parse_dates=['date'])
        print(self.df)
        self.attribute = self.df.columns  # 属性
        print("attribute=", self.attribute)

        self.yy = ['total_deaths', 'new_deaths', 'total_cases', 'new_cases',
                   'total_cases_per_million', 'new_cases_per_million',
                   'total_deaths_per_million', 'new_deaths_per_million', ]  # 取出来的数据类型
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化模型,归一化,反归一化要用的
        self.a = 'total_cases'
        self.name = 'AFG'
        self.preds = 0

    def go(self):
        self.division()

        self.describe()

        self.Normaldistribution()

        self.FuturePrediction()

    def division(self):  # 选取要分析的数据
        # 分割数据集
        self.df = self.df.loc[self.df['iso_code'] == self.name, self.yy]

        # 数据表格对齐
        pd.set_option('display.unicode.east_asian_width', True)

        # 删除空数据
        self.df.dropna(inplace=True)

    def describe(self):  # 描述性统计分析
        print("描述性统计分析:\n", self.df.describe())

    def Normaldistribution(self):  # 正太分布变换
        self.df[self.a] = np.log1p(self.df[self.a])  # 对数变换

    def FuturePrediction(self):
        # 选取数据
        # stock_week = self.df['total_cases'].resample('W-MON').mean()
        stock_week = self.df[self.a].resample('D').mean()
        stock_train = stock_week['2020-1-1': '2022-3-11']
        # print("stock_train\n", stock_train)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        # self.df.plot(figsize=(12, 8))
        # plt.legend(bbox_to_anchor=(1.25, 0.5))
        # plt.title("原数据图形")
        # sns.despine()

        # 求平稳性，原假设：单位根存在，序列不平稳，若p-value<0.05则拒绝原假设，说明稳定
        from statsmodels.tsa.stattools import adfuller
        d = 0
        ad = adfuller(stock_train)
        print("平稳性", ad)

        # 求参数d,差分法,差分只用于求参数d，不需要对真实的数据中进行差分
        births_diff = stock_train.diff()  # 参数为阶数，默认为1
        births_diff = births_diff.dropna()
        # plt.figure()
        # plt.plot(births_diff)
        # plt.title('一阶差分')
        # plt.show()

        ad = adfuller(births_diff)
        print("平稳性", ad)
        if ad[1] < 0.05:
            d = 1

        births_diff2 = births_diff.diff()  # 参数为阶数，默认为1
        births_diff2 = births_diff2.dropna()
        # plt.figure()
        # plt.plot(births_diff2)
        # plt.title('二阶差分')
        # plt.show()

        ad = adfuller(births_diff2)
        print("平稳性", ad)
        if ad[1] < 0.05:
            d = 2

        # 求参数p、q
        # 画acf和pacf
        # acf = plot_acf(births_diff2, lags=20)
        # plt.title("ACF")
        # acf.show()
        #
        # pacf = plot_pacf(births_diff2, lags=20)
        # plt.title("PACF")
        # pacf.show()

        # 评估
        # AIC = sm.tsa.arma_order_select_ic(stock_train, max_ar=7, max_ma=7, ic='aic')['aic_min_order']
        BIC = sm.tsa.arma_order_select_ic(stock_train, max_ar=7, max_ma=7, ic='bic')['bic_min_order']
        # print('the AIC is{},\nthe BIC is{}'.format(AIC, BIC))
        p = BIC[0]
        q = BIC[1]

        # # 循环找最好的参数,热力图法
        # p_min = 0
        # q_min = 0
        # p_max = 10
        # q_max = 10
        # d_min = 0
        # d_max = 10
        # # 创建Dataframe,以BIC准则
        # results_aic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
        #                            columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
        # # itertools.product 返回p,q中的元素的笛卡尔积的元组
        # for p, d, q in itertools.product(range(p_min, p_max + 1),
        #                                  range(d_min, d_max + 1), range(q_min, q_max + 1)):
        #     if p == 0 and q == 0:
        #         results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        #         continue
        #     try:
        #         model = sm.tsa.ARIMA(stock_train, order=(p, d, q))
        #         results = model.fit()
        #         # 返回不同pq下的model的BIC值
        #         results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
        #     except:
        #         continue
        # results_aic = results_aic[results_aic.columns].astype(float)
        # print("results_aic", results_aic)  # 热力矩阵
        # # 画热力图，选最小的值所在的坐标(p,q),aic和bic都有，且结果不一定相同
        # fig, ax = plt.subplots(figsize=(10, 8))
        # ax = sns.heatmap(results_aic,
        #                  # mask=results_aic.isnull(),
        #                  ax=ax,
        #                  annot=True,  # 将数字显示在热力图上
        #                  fmt='.2f',
        #                  )
        # ax.set_title('AIC')
        # plt.show()

        # 建立ARIMA模型
        model = sm.tsa.arima.ARIMA(stock_train, order=(p, d, q), freq='D')
        # model = sm.tsa.arima.ARIMA(stock_train, order=(p, d, q), freq='W-MON')
        result = model.fit()

        # 预测
        print("开始预测\n")
        pred = result.predict('2022-3-11', '2022-3-18', dynamic=True)
        pred = pd.DataFrame(pred)

        pred = np.expm1(pred)  # 这里要进行对数变换的反操作

        print("self.df[self.a]", np.expm1(self.df[self.a]))
        pred.iloc[0, 0] = np.expm1(self.df[self.a])[-1]  # 变第一个值
        print("pred=", pred)
        stock_week = np.expm1(stock_week)

        plt.figure(figsize=(6, 6))
        plt.plot(pred)
        plt.plot(stock_week)
        plt.savefig(f'结果文件\\{"ARIMA"}\\predict-data1{self.name}.PNG')
        # plt.show()
        pred.to_csv(f'结果文件\\{"ARIMA"}\\predict-data1{self.name}.csv', index=False, encoding="utf-8_sig")

        # 模型评估
        # result.plot_diagnostics(figsize=(16, 8))
        # plt.show()

        self.preds = pred


if __name__ == '__main__':
    l = ['GBR']
    for l1 in l:
        data = ARIMAprocessing()
        data.name = l1
        data.a = 'total_cases'
        # data.a = 'total_deaths'
        data.go()
