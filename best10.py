import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

'''       
        new_cases_smoothed_per_million:每 1,000,000 人中新确诊的 COVID-19 病例（7 天平滑）。包括报告的可能病例。
        new_deaths_smoothed_per_million:每 1,000,000 人中归因于 COVID-19 的新死亡（7 天平滑）。包括可能的死亡人数。
        icu_patients_per_million:每 1,000,000 人中特定日期在重症监护病房 (ICU) 中的 COVID-19 患者人数
        hosp_patients_per_million:每 1,000,000 人中某一天住院的 COVID-19 患者人数
        new_tests_smoothed_per_thousand:每 1,000 人的 COVID-19 新测试（7 天平滑）
        positive_rate:COVID-19 检测呈阳性的比例，以 7 天滚动平均值的形式给出（这是 tests_per_case 的倒数）
        new_vaccinations_smoothed_per_million:总人口中每 1,000,000 人接种的新 COVID-19 疫苗接种剂量（7 天平滑）
        new_people_vaccinated_smoothed_per_hundred:总人口中每 100 人每天接受第一剂疫苗（7 天平滑）的人数
        stringency_index:政府响应严格度指数：基于 9 个响应指标的综合衡量指标，包括学校停课、工作场所关闭和旅行禁令，重新调整为从 0 到 100 的值（100 = 最严格的响应）
        population_density:人口数除以土地面积，以平方公里为单位，
        aged_65_older:65 岁及以上的人口比例，最近一年可用
        extreme_poverty:生活在极端贫困中的人口比例，自 2010 年以来                         
        hospital_beds_per_thousand:每 1,000 人的病床数，自 2010 年以来
        human_development_index:衡量人类发展三个基本方面平均成就的综合指数——长寿和健康的生活、知识和体面的生活水平。2019 年的值      
        excess_mortality_cumulative:自2020 年 1 月 1 日以来的累计死亡人数与基于往年同期的累计预计死亡人数之间的百分比差异。 
        excess_mortality:年每周或每月报告的死亡人数与基于前几年的同期预计死亡人数之间的百分比差异。
        excess_mortality_cumulative_per_million:自 2020 年 1 月 1 日以来报告的死亡人数与基于前几年的同期预测死亡人数之间的累积差异，每百万人。
'''


def normalization(dataset, yy):  # 归一化
    for y in yy:
        scaler = MinMaxScaler()  # 归一化模型,归一化,反归一化要用的
        # 注意，下面的归一化别写错了，因为要有时间序列，和只有数据的归一化不同，reshape中-1是表示不论多少行，自动统计，1表示一列
        dataset[y] = scaler.fit_transform(dataset[y].values.reshape(-1, 1))
        print(y, '\n:', dataset[y])
    print("归一化后:\n", dataset)


def best10(start, end):
    dataset = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date', parse_dates=['date'])

    weight = {'new_cases_smoothed_per_million': -8,
              'new_deaths_smoothed_per_million': -7,
              'icu_patients_per_million': -9,
              'hosp_patients_per_million': -10,
              'new_tests_smoothed_per_thousand': 5,
              'positive_rate': -5,
              'new_vaccinations_smoothed_per_million': 9,
              'new_people_vaccinated_smoothed_per_hundred': 9,
              'stringency_index': 10,
              'population_density': 3,
              'aged_65_older': 3,
              'extreme_poverty': 5,
              'hospital_beds_per_thousand': 3,
              'human_development_index': 3,
              'excess_mortality_cumulative': -10,
              'excess_mortality': -10,
              'excess_mortality_cumulative_per_million': -10
              }  # 权重

    result = {}  # 结果
    # result1 = {}  # 结果

    normalization(dataset, weight.keys())  # 归一化

    # 获取所有国家,ds来存储
    ds = dataset['iso_code']
    ds.drop_duplicates(keep='first', inplace=True)  # 删除重复行,去除重复的国家
    for dname in ds:
        # print("当前国家:", dname)
        if 'OWID' not in dname:
            data = dataset.loc[dataset['iso_code'] == dname, weight.keys()]
            result[dname] = 0
            for at in weight.keys():
                # print(at)
                stock_day = data[at].resample('D').mean()
                stock_train = stock_day[start: end]  # 按照日期取出
                result[dname] += stock_train.mean() * weight[at]
                if dname in ['CHN', 'JPN', 'GRC']:
                    print("当前国家:", dname)
                    print(at, ':', stock_train.mean() * weight[at])
        # else:
        #     data = dataset.loc[dataset['iso_code'] == dname, weight.keys()]
        #     result1[dname] = 0
        #     for at in weight.keys():
        #         # print(at)
        #         stock_day = data[at].resample('D').mean()
        #         stock_train = stock_day[start: end]  # 按照日期取出
        #         result1[dname] += stock_train.mean() * weight[at]

    # 结果排序
    ri = sorted(result.items(), key=lambda x: x[1])[-10:]
    print(ri)

    # ri1 = sorted(result1.items(), key=lambda x: x[1])[-10:]
    # print(ri1)

    # 绘图
    x = []
    y = []
    for r in ri:
        x.append(r[0])
        y.append(round(r[1], 2))
    plt.figure(figsize=(6, 6))
    plt.bar(range(len(x)), y, width=0.4, color="yellow", edgecolor='black')
    plt.xticks(range(len(x)), x)
    for a, b in zip(range(len(x)), y):  # 柱子上面显示值
        plt.text(a, b, b, ha='center', va='bottom')
    plt.title("疫情治理最好的10个国家:")
    plt.savefig('结果文件\\best10.png')
    plt.show()

    # x = []
    # y = []
    # for r in ri1:
    #     x.append(r[0])
    #     y.append(r[1])
    # plt.figure(figsize=(6, 6))
    # plt.bar(range(len(x)), y, width=0.4, color="yellow", edgecolor='black')
    # plt.xticks(range(len(x)), x)
    # for a, b in zip(range(len(x)), y):  # 柱子上面显示值
    #     plt.text(a, b + 1, b, ha='center', va='bottom')
    # plt.title("OWID数据中最好10个数据")
    # plt.show()


if __name__ == '__main__':
    start = '2020-1-1'
    end = '2022-3-11'
    best10(start, end)
