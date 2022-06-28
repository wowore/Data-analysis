from pyecharts.charts import Bar
from pyecharts import options as opts
import os


class Draw_Bar:
    def __init__(self, time):
        self.time = time
        self.path1 = f'./Pyechart/Bar/Data_{self.time}'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        self.path2 = self.path1 + '/PerProvinceBar'
        if not os.path.exists(self.path2):
            os.makedirs(self.path2)

    def to_bar_china(self, area, curedRate, diedRate, update_time):
        bar1 = (
            Bar(init_opts=opts.InitOpts(width='800px', height='800px'))
                .add_xaxis(area)  # x轴列表样式
                .add_yaxis("治愈率", curedRate, color="#2f4554")  # y轴列表样式
                # .add_yaxis("死亡率", diedRate, color='#c23531', stack='stack1')  # y轴列表样式
                .set_global_opts(
                title_opts=opts.TitleOpts(title=f"各国疫情直方图", subtitle=f'截止{update_time} 各省治愈率情况', pos_left="center"
                                          ),
                # title_opts标题
                datazoom_opts=opts.DataZoomOpts(),
                legend_opts=opts.LegendOpts(is_show=True, pos_bottom='1px'),
                xaxis_opts=opts.AxisOpts(name='省'
                                         , axislabel_opts=opts.LabelOpts(font_size=10, rotate=45  # 字旋转的角度
                                                                         )  ##坐标轴标签的格式配置
                                         )
            )
        )

        bar1.render(self.path1 + f'/ChinaCuredBar.html')
        bar2 = (
            Bar(init_opts=opts.InitOpts(width='800px', height='800px'))
                .add_xaxis(area)  # x轴列表样式
                # .add_yaxis("治愈率", curedRate, color="#2f4554")  # y轴列表样式
                .add_yaxis("死亡率", diedRate, color='#c23531')  # y轴列表样式
                .set_global_opts(
                title_opts=opts.TitleOpts(title=f"各国疫情直方图", subtitle=f'截止{update_time} 各省死亡率情况', pos_left="center"),
                # title_opts标题
                datazoom_opts=opts.DataZoomOpts(),
                legend_opts=opts.LegendOpts(is_show=True, pos_bottom='1px'),
                xaxis_opts=opts.AxisOpts(name='省'
                                         , axislabel_opts=opts.LabelOpts(font_size=10, rotate=45  # 字旋转的角度
                                                                         )  ##坐标轴标签的格式配置
                                         )
            )
        )

        bar2.render(self.path1 + f'/ChinaDiedBar.html')

    def to_bar_province(self, area, curedRate, diedRate, province, update_time):
        bar1 = (
            Bar(init_opts=opts.InitOpts(width='800px', height='800px'))
                .add_xaxis(area)  # x轴列表样式
                .add_yaxis("治愈率", curedRate, color="#2f4554")  # y轴列表样式
                # .add_yaxis("死亡率", diedRate, color='#c23531', stack='stack1')  # y轴列表样式
                .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{province}疫情直方图", subtitle=f'截止{update_time} {province}治愈率情况',
                                          pos_left="center"),
                # title_opts标题
                datazoom_opts=opts.DataZoomOpts(),
                legend_opts=opts.LegendOpts(is_show=True, pos_bottom='1px'),
                xaxis_opts=opts.AxisOpts(name='市'
                                         , axislabel_opts=opts.LabelOpts(font_size=10, rotate=45  # 字旋转的角度
                                                                         )  ##坐标轴标签的格式配置
                                         )
            )
        )

        bar1.render(self.path2 + f'/{province}治愈率直方图.html')
        bar2 = (
            Bar(init_opts=opts.InitOpts(width='800px', height='800px'))
                .add_xaxis(area)  # x轴列表样式
                # .add_yaxis("治愈率", curedRate, color="#2f4554")  # y轴列表样式
                .add_yaxis("死亡率", diedRate, color='#c23531')  # y轴列表样式
                .set_global_opts(
                title_opts=opts.TitleOpts(title=f"{province}疫情直方图", subtitle=f'截止{update_time} {province}死亡率情况',
                                          pos_left="center"),
                # title_opts标题
                datazoom_opts=opts.DataZoomOpts(),
                legend_opts=opts.LegendOpts(is_show=True, pos_bottom='1px'),
                xaxis_opts=opts.AxisOpts(name='市'
                                         , axislabel_opts=opts.LabelOpts(font_size=10, rotate=45  # 字旋转的角度
                                                                         )  ##坐标轴标签的格式配置
                                         )
            )
        )

        bar2.render(self.path2 + f'/{province}死亡率直方图.html')
