from pyecharts import options as opts
from pyecharts.charts import WordCloud
import os

from pyecharts.globals import SymbolType


class Draw_WordCloud:
    def __init__(self, time):
        self.time = time
        self.path1 = f'./Pyechart/WordCloud/Data_{self.time}'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        self.path2 = self.path1 + '/PerProvinceWordCloud'
        if not os.path.exists(self.path2):
            os.makedirs(self.path2)
        self.width = '761px'
        self.height = '600px'

    def to_wc_province(self, area, variate, province, update_time):
        pass

    def to_wc_china(self, area, variate, update_time):
        pass

    def to_wc_world(self, area, variate, update_time):
        wc = WordCloud(init_opts=opts.InitOpts(width=self.width, height=self.height))
        wc.add("累计确诊人数", [(a, b) for a, b in zip(area, variate)],
               textstyle_opts=opts.TextStyleOpts(font_family="cursive"), shape=SymbolType.DIAMOND)
        wc.set_global_opts(
            title_opts=opts.TitleOpts(title=f"世界疫情词云图", subtitle=f'截止{update_time} 疫情分布情况',
                                      pos_left='center', pos_top='10px'),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        wc.render(self.path1 + f'/WorldWordCloud.html')
