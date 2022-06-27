import numpy as np
import pandas as pd
import os


def qvpath(path):
    g = []
    for dirpath, dirnames, filenames in os.walk(path):
        print("本文件路径:", dirpath, "文件夹名:", dirnames, "文件名:", filenames)
        for fi in filenames:
            g.append(dirpath + '\\' + fi)
    return g


class cleaning:
    def __init__(self):
        p = '数据文件'
        qvp = qvpath(p)
        for q in qvp:
            self.df = pd.read_csv(q)
            self.Deleteduplicatelines()
            self.Missingvaluefill()
            self.df.to_csv(q, encoding="utf-8_sig")  # 重新存入

    def Deleteduplicatelines(self):  # 删除重复行
        self.df.drop_duplicates(keep='first', inplace=True)  # 删除重复行
        self.df.reset_index(drop=True)  # 重新建立编号索引,drop用于把之前的index删除掉

    def Missingvaluefill(self):  # 缺失值填充
        for df1 in self.df:
            self.df[df1] = self.df[df1].replace('-', np.NaN)  # 将错误的值转换成空值
            # self.df[df1].fillna(self.df[df1].median(), inplace=True)  # 将空的位置变成中位数
            self.df[df1].fillna(0, inplace=True)  # 将空的位置变成0
            self.df[df1] = self.df[df1].astype('object')  # 强制转换


if __name__ == '__main__':
    c = cleaning()
