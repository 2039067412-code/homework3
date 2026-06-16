import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# ==================== 任务1 数据预处理 ====================
# 1.1 读取数据集
# 从本地指定路径读取CSV格式的公交刷卡原始数据集，加载为pandas的DataFrame结构
data_path = r"E:\中山大学\作业\python\homework3\ICData.csv"
df = pd.read_csv(data_path)

print("=" * 60)
print("【任务1 数据预处理】")
print("\n数据集前5行：")
print(df.head())
print("\n数据集基本信息：")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
print("各列数据类型：")
print(df.dtypes)

# 1.2 时间解析：转换为datetime类型，提取小时字段
# ========== 核心注释：时间解析 ==========
# 使用pd.to_datetime将CSV中字符串格式的时间戳，转换为pandas内置的datetime64标准时间类型
# 转换后才能支持按小时、分钟等任意时间粒度的筛选、聚合与计算，是时间序列分析的基础步骤
df['交易时间'] = pd.to_datetime(df['交易时间'])
# 通过datetime专属的dt访问器，从完整时间戳中提取小时分量（取值范围0-23的整数）
# 新增为hour独立列，用于后续分时统计、早晚高峰时段筛选等分析
df['hour'] = df['交易时间'].dt.hour

# 1.3 构造衍生字段：搭乘站点数，删除异常记录
# 对同一条记录的下车站点与上车站点做差后取绝对值，得到该次乘车实际经过的站点数量
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()
# 统计搭乘站点数为0的异常记录总数（上下车为同一站点属于无效乘车数据）
abnormal_num = (df['ride_stops'] == 0).sum()
# 过滤掉所有站点数为0的异常记录，重置索引保证行号连续
df = df[df['ride_stops'] != 0].reset_index(drop=True)
print(f"\n构造ride_stops后删除异常记录(ride_stops==0)行数: {abnormal_num}")

# 1.4 缺失值检查与处理
print("\n各列缺失值数量：")
# 逐列统计空值/缺失值的数量，检查数据完整性
missing_val = df.isnull().sum()
print(missing_val)
# 删除包含任意缺失值的行，确保后续计算不会因空值报错
df = df.dropna().reset_index(drop=True)
print(f"删除缺失值后剩余有效记录数: {df.shape[0]}")

# 预筛选上车刷卡记录（刷卡类型=0），后续所有分析任务均基于上车客流数据
df_board = df[df['刷卡类型'] == 0].copy()
print(f"\n有效上车刷卡记录总数: {df_board.shape[0]}")

