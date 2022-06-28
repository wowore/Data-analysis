from pyecharts import options as opts
from pyecharts.charts import Map
import os


class Draw_map():
    # relativeTime为发布的时间,传入时间戳字符串
    # def get_time(self):
    # relativeTime = int(relativeTime)
    # return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(relativeTime))

    def __init__(self, time):
        self.time = time
        self.path1 = f'./Pyechart/Map/Data_{self.time}'
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        self.path2 = self.path1 + '/PerProvinceMap'
        if not os.path.exists(self.path2):
            os.makedirs(self.path2)
        self.name_map = {
            'Singapore Rep.': '新加坡',
            'Dominican Rep.': '多米尼加',
            'Palestine': '巴勒斯坦',
            'Bahamas': '巴哈马',
            'Timor-Leste': '东帝汶',
            'Afghanistan': '阿富汗',
            'Guinea-Bissau': '几内亚比绍',
            "Côte d'Ivoire": '科特迪瓦',
            'Siachen Glacier': '锡亚琴冰川',
            "Br. Indian Ocean Ter.": '英属印度洋领土',
            'Angola': '安哥拉',
            'Albania': '阿尔巴尼亚',
            'United Arab Emirates': '阿联酋',
            'Argentina': '阿根廷',
            'Armenia': '亚美尼亚',
            'French Southern and Antarctic Lands': '法属南半球和南极领地',
            'Australia': '澳大利亚',
            'Austria': '奥地利',
            'Azerbaijan': '阿塞拜疆',
            'Burundi': '布隆迪',
            'Belgium': '比利时',
            'Benin': '贝宁',
            'Burkina Faso': '布基纳法索',
            'Bangladesh': '孟加拉国',
            'Bulgaria': '保加利亚',
            'The Bahamas': '巴哈马',
            'Bosnia and Herz.': '波斯尼亚和黑塞哥维那',
            'Belarus': '白俄罗斯',
            'Belize': '伯利兹',
            'Bermuda': '百慕大',
            'Bolivia': '玻利维亚',
            'Brazil': '巴西',
            'Brunei': '文莱',
            'Bhutan': '不丹',
            'Botswana': '博茨瓦纳',
            'Central African Rep.': '中非共和国',
            'Canada': '加拿大',
            'Switzerland': '瑞士',
            'Chile': '智利',
            'China': '中国',
            'Ivory Coast': '象牙海岸',
            'Cameroon': '喀麦隆',
            'Dem. Rep. Congo': '刚果（金）',
            'Congo': '刚果（布）',
            'Colombia': '哥伦比亚',
            'Costa Rica': '哥斯达黎加',
            'Cuba': '古巴',
            'N. Cyprus': '北塞浦路斯',
            'Cyprus': '塞浦路斯',
            'Czech Rep.': '捷克',
            'Germany': '德国',
            'Djibouti': '吉布提',
            'Denmark': '丹麦',
            'Algeria': '阿尔及利亚',
            'Ecuador': '厄瓜多尔',
            'Egypt': '埃及',
            'Eritrea': '厄立特里亚',
            'Spain': '西班牙',
            'Estonia': '爱沙尼亚',
            'Ethiopia': '埃塞俄比亚',
            'Finland': '芬兰',
            'Fiji': '斐',
            'Falkland Islands': '福克兰群岛',
            'France': '法国',
            'Gabon': '加蓬',
            'United Kingdom': '英国',
            'Georgia': '格鲁吉亚',
            'Ghana': '加纳',
            'Guinea': '几内亚',
            'Gambia': '冈比亚',
            'Guinea Bissau': '几内亚比绍',
            'Eq. Guinea': '赤道几内亚',
            'Greece': '希腊',
            'Greenland': '格陵兰岛',
            'Guatemala': '危地马拉',
            'French Guiana': '法属圭亚那',
            'Guyana': '圭亚那',
            'Honduras': '洪都拉斯',
            'Croatia': '克罗地亚',
            'Haiti': '海地',
            'Hungary': '匈牙利',
            'Indonesia': '印度尼西亚',
            'India': '印度',
            'Ireland': '爱尔兰',
            'Iran': '伊朗',
            'Iraq': '伊拉克',
            'Iceland': '冰岛',
            'Israel': '以色列',
            'Italy': '意大利',
            'Jamaica': '牙买加',
            'Jordan': '约旦',
            'Japan': '日本',
            'Kazakhstan': '哈萨克斯坦',
            'Kenya': '肯尼亚',
            'Kyrgyzstan': '吉尔吉斯斯坦',
            'Cambodia': '柬埔寨',
            'Korea': '韩国',
            'Kosovo': '科索沃',
            'Kuwait': '科威特',
            'Lao PDR': '老挝',
            'Lebanon': '黎巴嫩',
            'Liberia': '利比里亚',
            'Libya': '利比亚',
            'Sri Lanka': '斯里兰卡',
            'Lesotho': '莱索托',
            'Lithuania': '立陶宛',
            'Luxembourg': '卢森堡',
            'Latvia': '拉脱维亚',
            'Morocco': '摩洛哥',
            'Moldova': '摩尔多瓦',
            'Madagascar': '马达加斯加',
            'Mexico': '墨西哥',
            'Macedonia': '马其顿',
            'Mali': '马里',
            'Myanmar': '缅甸',
            'Montenegro': '黑山',
            'Mongolia': '蒙古国',
            'Mozambique': '莫桑比克',
            'Mauritania': '毛里塔尼亚',
            'Malawi': '马拉维',
            'Malaysia': '马来西亚',
            'Namibia': '纳米比亚',
            'New Caledonia': '新喀里多尼亚',
            'Niger': '尼日尔',
            'Nigeria': '尼日利亚',
            'Nicaragua': '尼加拉瓜',
            'Netherlands': '荷兰',
            'Norway': '挪威',
            'Nepal': '尼泊尔',
            'New Zealand': '新西兰',
            'Oman': '阿曼',
            'Pakistan': '巴基斯坦',
            'Panama': '巴拿马',
            'Peru': '秘鲁',
            'Philippines': '菲律宾',
            'Papua New Guinea': '巴布亚新几内亚',
            'Poland': '波兰',
            'Puerto Rico': '波多黎各',
            'Dem. Rep. Korea': '朝鲜',
            'Portugal': '葡萄牙',
            'Paraguay': '巴拉圭',
            'Qatar': '卡塔尔',
            'Romania': '罗马尼亚',
            'Russia': '俄罗斯',
            'Rwanda': '卢旺达',
            'W. Sahara': '西撒哈拉',
            'Saudi Arabia': '沙特阿拉伯',
            'Sudan': '苏丹',
            'S. Sudan': '南苏丹',
            'Senegal': '塞内加尔',
            'Solomon Is.': '所罗门群岛',
            'Sierra Leone': '塞拉利昂',
            'El Salvador': '萨尔瓦多',
            'Somaliland': '索马里兰',
            'Somalia': '索马里',
            'Serbia': '塞尔维亚',
            'Suriname': '苏里南',
            'Slovakia': '斯洛伐克',
            'Slovenia': '斯洛文尼亚',
            'Sweden': '瑞典',
            'Swaziland': '斯威士兰',
            'Syria': '叙利亚',
            'Chad': '乍得',
            'Togo': '多哥',
            'Thailand': '泰国',
            'Tajikistan': '塔吉克斯坦',
            'Turkmenistan': '土库曼斯坦',
            'East Timor': '东帝汶',
            'Trinidad and Tobago': '特里尼达和多巴哥',
            'Tunisia': '突尼斯',
            'Turkey': '土耳其',
            'Tanzania': '坦桑尼亚',
            'Uganda': '乌干达',
            'Ukraine': '乌克兰',
            'Uruguay': '乌拉圭',
            'United States': '美国',
            'Uzbekistan': '乌兹别克斯坦',
            'Venezuela': '委内瑞拉',
            'Vietnam': '越南',
            'Vanuatu': '瓦努阿图',
            'West Bank': '西岸',
            'Yemen': '也门',
            'South Africa': '南非',
            'Zambia': '赞比亚',
            'Zimbabwe': '津巴布韦',
            'Comoros': '科摩罗'
        }

    def get_colour(self, a, b, c):
        result = '#' + ''.join(map((lambda x: "%02x" % x), (a, b, c)))
        return result.upper()

    '''
    参数说明——area：地级市 variate：对应的疫情数据 province：省份（不含省字）
    '''

    # def to_map_city(self, area, variate, province, update_time):
    #     pieces = [
    #         {"max": 99999999, "min": 10000, "label": "≥10000", "color": self.get_colour(102, 2, 8)},
    #         {"max": 9999, "min": 1000, "label": "1000-9999", "color": self.get_colour(140, 13, 13)},
    #         {"max": 999, "min": 500, "label": "500-999", "color": self.get_colour(204, 41, 41)},
    #         {"max": 499, "min": 100, "label": "100-499", "color": self.get_colour(255, 123, 105)},
    #         {"max": 99, "min": 50, "label": "50-99", "color": self.get_colour(255, 170, 133)},
    #         {"max": 49, "min": 10, "label": "10-49", "color": self.get_colour(255, 202, 179)},
    #         {"max": 9, "min": 1, "label": "1-9", "color": self.get_colour(255, 228, 217)},
    #         {"max": 0, "min": 0, "label": "0", "color": self.get_colour(255, 255, 255)},
    #     ]
    #     c = (
    #         # 设置地图大小
    #         Map(init_opts=opts.InitOpts(width='800px', height='800px'))
    #             .add("累计确诊人数", [list(z) for z in zip(area, variate)], province, is_map_symbol_show=False)
    #             # 设置全局变量  is_piecewise设置数据是否连续，split_number设置为分段数，pices可自定义数据分段
    #             # is_show设置是否显示图例
    #             .set_global_opts(
    #             title_opts=opts.TitleOpts(title="%s地区疫情地图分布" % (province),
    #                                       subtitle='截止%s  %s省疫情分布情况' % (update_time, province), pos_left="center",
    #                                       pos_top="10px"),
    #             legend_opts=opts.LegendOpts(is_show=False),
    #             visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True,
    #                                               pieces=pieces,
    #                                               ),
    #         )
    #             .render("./map/china/{}疫情地图.html".format(province))
    #     )
    def to_map_province(self, area, variate, province, update_time):
        map = Map(
            init_opts=opts.InitOpts(width="800px", height="800px", page_title=f"中国{province}确诊人数"))
        map.add("累计确诊人数", [list(z) for z in zip(area, variate)], is_map_symbol_show=True,
                maptype=province, label_opts=opts.LabelOpts(is_show=False),)  # 地图区域颜色
        map.set_global_opts(
            title_opts=opts.TitleOpts(title=f"{province}疫情地图分布", subtitle=f'截止{update_time} 中国疫情分布情况',
                                      pos_left="center",
                                      pos_top="10px"), legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True,
                                              pieces=[
                                                  {"max": 99999999, "min": 10000, "label": ">10000", "color": "#8A0808"},
                                                  {"max": 9999, "min": 1000, "label": "1000-9999", "color": "#B40404"},
                                                  {"max": 999, "min": 100, "label": "100-999", "color": "#DF0101"},
                                                  {"max": 99, "min": 10, "label": "10-99", "color": "#F78181"},
                                                  {"max": 9, "min": 1, "label": "1-9", "color": "#F5A9A9"},
                                                  {"max": 0, "min": 0, "label": "0", "color": "#FFFFFF"},
                                              ])
        )
        map.render(self.path2 + f'/中国{province}累计确诊人数.html')

    def to_map_china(self, area, variate, update_time):
        map = Map(init_opts=opts.InitOpts(width="800px", height="800px", page_title="中国累计确诊人数"))
        map.add("累计确诊人数", [list(z) for z in zip(area, variate)], is_map_symbol_show=False,
                maptype="china", label_opts=opts.LabelOpts(is_show=False))  # 地图区域颜色
        map.set_global_opts(
            title_opts=opts.TitleOpts(title="中国疫情地图分布", subtitle='截止%s 中国疫情分布情况' % (update_time), pos_left="center",
                                      pos_top="10px"), legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True,
                                              pieces=[
                                                  {"max": 99999999, "min": 10000, "label": ">10000", "color": "#8A0808"},
                                                  {"max": 9999, "min": 1000, "label": "1000-9999", "color": "#B40404"},
                                                  {"max": 999, "min": 100, "label": "100-999", "color": "#DF0101"},
                                                  {"max": 99, "min": 10, "label": "10-99", "color": "#F78181"},
                                                  {"max": 9, "min": 1, "label": "1-9", "color": "#F5A9A9"},
                                                  {"max": 0, "min": 0, "label": "0", "color": "#FFFFFF"},
                                              ])
        )
        map.render(self.path1 + '/ChinaMap.html')

    def to_map_world(self, country, confirmed, update_time):
        map = Map(init_opts=opts.InitOpts(width="800px", height="800px", page_title="全世界累计确诊人数"))
        map.add("累计确诊人数", [list(z) for z in zip(country, confirmed)], is_map_symbol_show=False, name_map=self.name_map,
                maptype="world", label_opts=opts.LabelOpts(is_show=False))  # 地图区域颜色
        map.set_global_opts(
            title_opts=opts.TitleOpts(title="世界疫情地图分布", subtitle='截止%s 世界疫情分布情况' % (update_time), pos_left="center",
                                      pos_top="10px"), legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(max_=1000000000, is_piecewise=True,
                                              pieces=[
                                                  {"max": 999999999, "min": 10000001,
                                                   "label": ">10000000",
                                                   "color": "#8A0808"},
                                                  {"max": 9999999, "min": 1000000,
                                                   "label": "1000000-9999999",
                                                   "color": "#B40404"},
                                                  {"max": 999999, "min": 100000,
                                                   "label": "100000-999999",
                                                   "color": "#DF0101"},
                                                  {"max": 99999, "min": 10000, "label": "10000-99999",
                                                   "color": "#F78181"},
                                                  {"max": 9999, "min": 1, "label": "1-9999",
                                                   "color": "#F5A9A9"},
                                                  {"max": 0, "min": 0, "label": "0",
                                                   "color": "#FFFFFF"},
                                              ])
        )
        map.render(self.path1 + '/WorldMap.html')
