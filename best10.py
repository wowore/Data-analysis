import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def best10(start, end):
    dataset = pd.read_csv("数据文件\\owid-covid-data.csv", encoding="utf-8_sig", index_col='date', parse_dates=['date'])

    weight = {'new_cases_per_million': -5, 'new_cases_smoothed_per_million': -5, 'total_deaths_per_million': -10,
              'new_deaths_per_million': -5, 'new_deaths_smoothed_per_million': -5, 'icu_patients_per_million': -9,
              'hosp_patients_per_million': -10, 'weekly_icu_admissions_per_million': -7,
              'weekly_hosp_admissions_per_million': -6, 'new_tests_per_thousand': 5,
              'new_tests_smoothed_per_thousand': 5,
              'positive_rate': -5, 'new_vaccinations': 10, 'new_vaccinations_smoothed': 10,
              'total_vaccinations_per_hundred': 10,
              'people_vaccinated_per_hundred': 9, 'people_fully_vaccinated_per_hundred': 8,
              'total_boosters_per_hundred': 10, 'new_vaccinations_smoothed_per_million': 9,
              'new_people_vaccinated_smoothed': 8, 'new_people_vaccinated_smoothed_per_hundred': 9,
              'stringency_index': 10, 'population_density': 3, 'median_age': 1, 'aged_65_older': 5, 'aged_70_older': 6,
              'gdp_per_capita': 5, 'extreme_poverty': 10, 'cardiovasc_death_rate': -6, 'diabetes_prevalence': -6,
              'female_smokers': -3, 'male_smokers': -4, 'handwashing_facilities': 8, 'hospital_beds_per_thousand': 8,
              'life_expectancy': 1, 'human_development_index': 10, 'excess_mortality_cumulative_absolute': -10,
              'excess_mortality_cumulative': -10, 'excess_mortality': -10,
              'excess_mortality_cumulative_per_million': -10
              }  # 权重
    result = {}  # 结果
    # result1 = {}  # 结果

    # 获取所有国家,ds来存储
    ds = dataset['iso_code']
    ds.drop_duplicates(keep='first', inplace=True)  # 删除重复行,去除重复的国家
    for dname in ds:
        print(dname)
        if 'OWID' not in dname:
            data = dataset.loc[dataset['iso_code'] == dname, weight.keys()]
            result[dname] = 0
            for at in weight.keys():
                # print(at)
                stock_day = data[at].resample('D').mean()
                stock_train = stock_day[start: end]  # 按照日期取出
                result[dname] += stock_train.mean() * weight[at]
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
        y.append(r[1])
    plt.figure(figsize=(6, 6))
    plt.bar(range(len(x)), y, width=0.4, color="yellow", edgecolor='black')
    plt.xticks(range(len(x)), x)
    for a, b in zip(range(len(x)), y):  # 柱子上面显示值
        plt.text(a, b + 1, b, ha='center', va='bottom')
    plt.title("疫情治理最好的10个国家:")
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
