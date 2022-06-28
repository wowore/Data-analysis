import json

import openpyxl
from wordcloud import WordCloud
from pyecharts import options as opts
# from pyecharts.charts import WordCloud

# 与即时显示图片相关的模块

import matplotlib.pyplot as plt  # 绘制图像的模块
import numpy as np
from PIL import Image
import os

# 读取数据
wb = openpyxl.load_workbook('Data/Data2022.05.26.xlsx')
sheet_names = wb.sheetnames
with open('Data/globalData2022.05.26.json', 'r') as f:
    data = f.read()
    data = json.loads(data)

frequency_out = {}
frequency_out['中国'] = float(data['中国'])
for each in sheet_names:
    if '洲' in each:
        ws = wb[each]
        for row in ws.values:
            if row[1] == "累计确诊":
                pass
            else:
                frequency_out[row[0]] = float(row[1])
    else:
        pass

# 以省份的确诊病例总数代表其出现的频率
frequency_in = {}

ws = wb['国内疫情']
for row in ws.values:
    if row[1] == "累计确诊":
        pass
    else:
        frequency_in[row[0]] = float(row[1])


def generate_pic(frequency, name):
    # 这里可以事先准备一张图片，可以用作背景
    background_image = np.array(Image.open('00001.jpg'))
    wordcloud = WordCloud(font_path="C:/Windows/Fonts/simhei.ttf",
                          background_color="white",
                          # mask=background_image,
                          width=1920, height=1080
                          )
    # 按照确诊病例数目生成词云
    wordcloud.generate_from_frequencies(frequency)
    wordcloud.to_file('%s.png' % name)


# 调用函数
generate_pic(frequency_in, '国内疫情')
os.startfile('国内疫情.png')
generate_pic(frequency_out, '国外疫情')
os.startfile('国外疫情.png')
