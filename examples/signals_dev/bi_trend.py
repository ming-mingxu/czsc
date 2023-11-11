# -*- coding: UTF-8 -*-
"""
@Project ：czsc 
@File    ：bi_trend.py
@Author  ：xuming
@Date    ：2023/9/13 20:46 
@describe: 
"""
import os
import sys

sys.path.insert(0, '..')
os.environ['czsc_verbose'] = '1'
os.environ['czsc_research_cache'] = r"E:\Download\CZSC投研数据"

from collections import OrderedDict
from czsc import CZSC, Direction
from czsc.utils import create_single_signal
from czsc.traders.base import check_signals_acc
from czsc.connectors import research

import numpy as np
from sklearn.linear_model import LinearRegression


def bi_trend_V230913(c: CZSC, **kwargs) -> OrderedDict:
    """辅助判断股票通道信号

    参数模板："{freq}_D{di}N{n}笔高低点趋势线V230913"

    **信号逻辑：**

    1. 白仪底分型停顿，认为是向下笔结束；反之，向上笔结束
    2. 底分型停顿：底分型后一根大阳线收盘在底分型的高点上方；反之，顶分型停顿

    **信号列表：**

    - Signal('15分钟_D0停顿分型_BE辅助V230106_看多_强_任意_0')
    - Signal('15分钟_D0停顿分型_BE辅助V230106_看空_强_任意_0')
    - Signal('15分钟_D0停顿分型_BE辅助V230106_看空_弱_任意_0')
    - Signal('15分钟_D0停顿分型_BE辅助V230106_看多_弱_任意_0')

    **Notes：**

    1. BE 是 Bi End 的缩写

    :param c: CZSC对象
    :return: 信号识别结果
    """
    di = int(kwargs.get("di", 4))
    assert di >= 2

    n = int(kwargs.get("n", 1))

    k1, k2, k3 = f"{c.freq.value}_D{di}N{n}_笔高低点趋势线V230913".split('_')
    v1 = "其他"

    if len(c.bi_list) < 3:
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    up_trend_price = np.array([x.high for x in c.bi_list[-di:] if x.direction == Direction.Up])
    up_trend_time = np.array([x.edt.timestamp() for x in c.bi_list[-di:] if x.direction == Direction.Up]).reshape(-1, 1)

    down_price = np.array([x.low for x in c.bi_list[-di:] if x.direction == Direction.Up])
    down_trend_time = np.array([x.edt.timestamp() for x in c.bi_list[-di:] if x.direction == Direction.Down]).reshape(
        -1, 1)

    model_up = LinearRegression()  # 创建线性回归模型
    model_down = LinearRegression()  # 创建线性回归模型
    model_up.fit(up_trend_time, up_trend_price)  # 训练模型
    model_down.fit(down_trend_time, down_price)  # 训练模型

    new_bar_data = np.array([c.bars_ubi[-n].dt.timestamp()]).reshape(-1, 1)  # 新的时间数据
    pre_up_price = model_up.predict(new_bar_data)
    pre_down_price = model_down.predict(new_bar_data)
    pre_mid_price = (pre_up_price + pre_down_price) / 2

    if pre_up_price <= pre_down_price:
        v1 = "观望"
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

    if c.bars_raw[-n].close >= pre_up_price:
        v1 = '上升趋势'
        v2 = '超强'
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1, v2=v2)
    elif pre_mid_price < c.bars_raw[-n].close < pre_up_price:
        v1 = '上升趋势'
        v2 = '强'
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1, v2=v2)
    elif pre_down_price < c.bars_raw[-n].close < pre_mid_price:
        v1 = '下降趋势'
        v2 = '强'
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1, v2=v2)
    elif c.bars_raw[-n].close <= pre_down_price:
        v1 = '下降趋势'
        v2 = '超强'
        return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1, v2=v2)

    return create_single_signal(k1=k1, k2=k2, k3=k3, v1=v1)

# def main():
#     symbols = research.get_symbols('A股主要指数')
#     bars = research.get_raw_bars(symbols[0], '15分钟', '20171101', '20210101', fq='前复权')
#
#     signals_config = [
#         {'name': bi_trend_V230913, 'freq': '日线', 'di': 2},
#     ]
#     check_signals_acc(bars, signals_config=signals_config)
#
#
# if __name__ == '__main__':
#     main()
symbols = research.get_symbols('A股主要指数')
bars = research.get_raw_bars(symbols[0], '15分钟', '20181101', '20210101', fq='前复权')

signals_config = [
    {'name': bi_trend_V230913, 'freq': '15分钟'},
    # {'name': bar_zdt_V230331, 'freq': '60分钟'},
]

if __name__ == '__main__':
    check_signals_acc(bars, signals_config=signals_config)