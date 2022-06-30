import pandas as pd
import LSTMp1
import ARIMA
import LSTM
import lightGBM


class allprocessing:
    def __init__(self):
        self.df = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date', parse_dates=['date'])
        self.attribute = self.df.columns  # 属性
        print("attribute=", self.attribute)

        self.yy = ['total_deaths', 'new_deaths', 'total_cases', 'new_cases',
                   'total_cases_per_million', 'new_cases_per_million',
                   'total_deaths_per_million', 'new_deaths_per_million', ]  # 取出来的数据类型
        self.shape = []  # 最相关的特征

        self.a = 'total_cases'
        self.name = 'AFG'

        self.path = '结果文件\\predict-data.csv'
        self.path1 = '结果文件\\predict-data1.csv'

    def division(self):
        # 分割数据集
        self.df = self.df.loc[self.df['iso_code'] == self.name, self.yy]

        # 数据表格对齐
        pd.set_option('display.unicode.east_asian_width', True)

        # 删除空数据
        self.df.dropna(inplace=True)

    def clean(self):
        f = open(self.path, 'w').close()

    def relevance(self):  # 相关性分析
        corr = self.df.corr()
        print("相关性分析：\n", corr)
        d = []
        for c in self.attribute:
            if c in self.yy and c != self.a:
                d.append((c, corr[self.a][c]))
        for sd in sorted(d, key=lambda x: x[1], reverse=True)[:3]:  # 选相关性最高的3个属性,且相关性要超过0.8才能作为特征
            if abs(sd[1]) >= 0.7:
                print(sd[0], "相关性=", sd[1])
                self.shape.append(sd[0])
        self.shape.append(self.a)
        print("特征\n", self.shape)
        # print(corr['total_cases'])
        # print(corr['total_cases']['total_cases'])
        # print(corr['total_cases'][0])两种访问方式

    def LSTMp1(self):
        self.path = f'结果文件\\{"LSTMp1"}\\predict-data{self.name}.csv'
        # 相关性分析
        self.relevance()
        # 分割数据集
        self.division()
        # 清空csv文件内容
        # self.clean()

        # 创建新的dataset
        datasets = pd.DataFrame()

        for s in self.shape:
            print("==========================================", s, "===============================================")
            lstmp1 = LSTMp1.LSTMprocessing()
            lstmp1.a = s
            lstmp1.yy = self.yy
            lstmp1.name = self.name
            lstmp1.go()
            dataset = lstmp1.preds

            dataset.columns = [s]
            pd.set_option('display.unicode.east_asian_width', True)
            print("看这里！！！！！！！！\n", dataset[s])

            datasets.insert(loc=0, column=s, value=dataset.iloc[:, [0]], allow_duplicates=False)
            # 添加新列：DataFrame.insert(loc, column, value,allow_duplicates = False)loc
            # 必要字段，int类型数据，表示插入新列的列位置，原来在该位置的列将向右移。 column	必要字段，插入新列的列名。 value
            # 必要字段，新列插入的值。如果仅提供一个值，将为所有行设置相同的值。可以是int，string，float等，甚至可以是series /值列表。 allow_duplicates
            # 布尔值，用于检查是否存在具有相同名称的列。默认为False，不允许与已有的列名重复。 dataset.insert(loc=0, column='wo', value=[1, 2, 3, 4, 5, 6,
            # 7, 8, 9, 10]) 存入文件
        # 为了把日期加入进去
        if 'date' not in datasets.columns:
            datasets.insert(loc=0, column='date', value=dataset.index, allow_duplicates=False)
        datasets.to_csv(self.path, index=False, encoding="utf-8_sig")

    def ARIMA(self):
        self.path = f'结果文件\\{"ARIMA"}\\predict-data{self.name}.csv'
        # 相关性分析
        self.relevance()
        self.division()
        # 清空csv文件内容
        # self.clean()

        # 创建新的dataset
        datasets = pd.DataFrame()

        for s in self.shape:
            print("==========================================", s, "===============================================")
            arima = ARIMA.ARIMAprocessing()
            arima.a = s
            arima.yy = self.yy
            arima.name = self.name
            arima.go()
            dataset = arima.preds

            dataset.columns = [s]
            pd.set_option('display.unicode.east_asian_width', True)
            print("看这里！！！！！！！！\n", dataset[s])

            datasets.insert(loc=0, column=s, value=dataset.iloc[:, [0]], allow_duplicates=False)
            # 添加新列：DataFrame.insert(loc, column, value,allow_duplicates = False)loc
            # 必要字段，int类型数据，表示插入新列的列位置，原来在该位置的列将向右移。 column	必要字段，插入新列的列名。 value
            # 必要字段，新列插入的值。如果仅提供一个值，将为所有行设置相同的值。可以是int，string，float等，甚至可以是series /值列表。 allow_duplicates
            # 布尔值，用于检查是否存在具有相同名称的列。默认为False，不允许与已有的列名重复。 dataset.insert(loc=0, column='wo', value=[1, 2, 3, 4, 5, 6,
            # 7, 8, 9, 10]) 存入文件
        # 为了把日期加入进去
        if 'date' not in datasets.columns:
            datasets.insert(loc=0, column='date', value=dataset.index, allow_duplicates=False)
        datasets.to_csv(self.path, index=False, encoding="utf-8_sig")

    def LSTM(self):
        self.path1 = f'结果文件\\{"LSTM"}\\predict-data1{self.name}.csv'
        lstm = LSTM.LSTMprocessing()
        lstm.a = self.a
        lstm.name = self.name
        lstm.yy = self.shape
        lstm.path = self.path
        lstm.go()
        lstm.Predict()
        # 为了把日期加入进去
        datasets = lstm.pred
        datasets.columns = [self.a]
        if 'date' not in datasets.columns:
            datasets.insert(loc=0, column='date', value=datasets.index, allow_duplicates=False)
        datasets.to_csv(self.path1, index=False, encoding="utf-8_sig")

    def GBM(self):
        self.path1 = f'结果文件\\{"GBM"}\\predict-data1{self.name}.csv'
        gbm = lightGBM.GBMprocessing()
        gbm.a = self.a
        gbm.name = self.name
        gbm.yy = self.shape
        gbm.path = self.path
        gbm.go()
        gbm.Predict()
        # 为了把日期加入进去
        datasets = gbm.pred
        datasets.columns = [self.a]
        if 'date' not in datasets.columns:
            datasets.insert(loc=0, column='date', value=datasets.index, allow_duplicates=False)
        datasets.to_csv(self.path1, index=False, encoding="utf-8_sig")


if __name__ == '__main__':
    # CHN,JPN,USA,DEU,韩国KOR，HKG,FRA,GBR
    l = ['CHN', 'FRA', 'GBR', 'JPN', 'USA', 'DEU', 'KOR', 'HKG', 'FRA', 'GBR']
    for l1 in l:
        gcp = allprocessing()
        gcp.yy = ['total_cases',
                  'new_cases',
                  'new_cases_smoothed',
                  'total_cases_per_million',
                  'new_cases_per_million',
                  'new_cases_smoothed_per_million',

                  'total_deaths',
                  'new_deaths',
                  'new_deaths_smoothed',
                  'total_deaths_per_million',
                  'new_deaths_per_million',
                  'new_deaths_smoothed_per_million',

                  'icu_patients',
                  'hosp_patients',
                  'weekly_hosp_admissions',
                  'weekly_icu_admissions',
                  'icu_patients_per_million',
                  'hosp_patients_per_million',
                  'weekly_icu_admissions_per_million',
                  'weekly_hosp_admissions_per_million',
                  'hospital_beds_per_thousand',

                  'total_tests_per_thousand',
                  'new_tests_per_thousand',
                  'total_tests',
                  'new_tests',
                  'new_tests_smoothed',
                  'new_tests_smoothed_per_thousand',
                  'tests_per_case',

                  'total_vaccinations',
                  'people_vaccinated',
                  'people_fully_vaccinated',
                  'total_boosters',
                  'new_vaccinations',
                  'new_vaccinations_smoothed',
                  'total_vaccinations_per_hundred',
                  'people_vaccinated_per_hundred',
                  'people_fully_vaccinated_per_hundred',
                  'total_boosters_per_hundred',
                  'new_vaccinations_smoothed_per_million',
                  'new_people_vaccinated_smoothed',
                  'new_people_vaccinated_smoothed_per_hundred',

                  'positive_rate',
                  'reproduction_rate',
                  'stringency_index',
                  'population',

                  'median_age',
                  'aged_65_older',
                  'aged_70_older',

                  'extreme_poverty',
                  'gdp_per_capita',
                  'human_development_index',

                  ]  # 取出来的数据类型
        # gcp.a = 'total_cases'
        gcp.a = 'new_cases_per_million'
        gcp.name = l1

        gcp.ARIMA()
        # gcp.LSTMp1()
        gcp.GBM()
        # gcp.LSTM()

    # gcp.a = 'total_cases'
    # gcp.name = 'CHN'
    # gcp.ARIMA()
    # # gcp.LSTMp1()
    #
    # # gcp.GBM()
    # gcp.LSTM()

    # # 获取所有国家
    # dataset = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig")
    # ds = dataset['iso_code']
    # ds.drop_duplicates(keep='first', inplace=True)  # 删除重复行
    # for dname in ds:
    #     print(dname)
