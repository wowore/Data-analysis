import sys

from PyQt5.QtWidgets import QApplication

import drawMap
import json
import run

map = drawMap.Draw_map()
# 格式
# map.to_map_china(['湖北'],['99999'],'1584201600')
# map.to_map_city(['荆门市'],['99999'],'湖北','1584201600')

# 获取数据
with open('Data/Data2022.05.26.json', 'r') as file:
    data = file.read()
    data = json.loads(data)
# print(data)

# 国外疫情获取
with open('Data/globalData2022.05.26.json', 'r') as file:
    data1 = file.read()
    data1 = json.loads(data1)

# 以省份为单位
with open('Data/chinaData2022.05.26.json', 'r') as file:
    data2 = file.read()
    data2 = json.loads(data2)
    print(data2)


# 中国疫情地图
def china_map(update_time):
    area = []
    confirmed = []
    for each in data2.items():
        print(each)
        area.append(each[0])
        confirmed.append(each[1])
    print(area)
    print(confirmed)
    map.to_map_china(area, confirmed, update_time)


# 23个省、5个自治区、4个直辖市、2个特别行政区 香港、澳门和台湾的subList为空列表，未有详情数据

# 省、直辖市疫情地图
def province_map(update_time):
    i = 0
    for each in data:
        # i += 1
        # print(f'{i}>>>>>>{each}')
        city = []
        confirmeds = []
        province = each['area']
        for each_city in each['subList']:
            city.append(each_city['city'] + "市")
            confirmeds.append(each_city['confirmed'])
            map.to_map_province(city, confirmeds, province, update_time)
        if province == '上海' or '北京' or '天津' or '重庆':
            for each_city in each['subList']:
                city.append(each_city['city'])
                confirmeds.append(each_city['confirmed'])
                map.to_map_province(city, confirmeds, province, update_time)


def world_map(update_time):
    area = []
    confirmed = []
    for item in data1.items():
        if item[0] == '钻石公主号邮轮':
            continue
        area.append(item[0])
        confirmed.append(item[1])
    map.to_map_world(area, confirmed, update_time)


# china_map('2022/5/25')
# # province_map('2022/5/25')
# world_map('2022/5/26')
# app = QApplication(sys.argv)
# ex = run.UI('WorldMap.html', '各国累计确诊人数')
# in2 = run.UI('ChinaMap.html', '中国累计确诊人数')
# ex.show()
# in2.show()
# sys.exit(app.exec_())
# province_map('2022/5/26')


