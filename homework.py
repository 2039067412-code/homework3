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


# ==================== 任务2 时间分布分析 ====================
print("\n" + "=" * 60)
print("【任务2 时间分布分析】")

# 2.1 numpy布尔索引统计早晚时段刷卡量
# 将pandas的小时列转换为numpy数组，满足题目要求使用numpy完成条件统计的规范
hour_array = df_board['hour'].values
total_swipes = len(hour_array)

# 早峰前：小时小于7
# 通过numpy布尔索引筛选出hour<7的元素，np.sum对布尔数组求和即得到符合条件的记录总数
before_peak_num = np.sum(hour_array < 7)
# 深夜时段：小时大于等于22
late_night_num = np.sum(hour_array >= 22)

# 计算两个时段刷卡量占全天总刷卡量的百分比
before_peak_ratio = before_peak_num / total_swipes * 100
late_night_ratio = late_night_num / total_swipes * 100

print("\n(a) 早晚时段刷卡量统计：")
print(f"早峰前(<7:00)刷卡量: {before_peak_num} 次，占全天比例: {before_peak_ratio:.2f}%")
print(f"深夜时段(≥22:00)刷卡量: {late_night_num} 次，占全天比例: {late_night_ratio:.2f}%")

# 2.2  matplotlib绘制24小时刷卡量柱状图
# 按小时分组统计刷卡量，reindex补全0-23所有小时，避免无数据的小时出现空缺
hour_counts = df_board.groupby('hour').size().reindex(range(24), fill_value=0)

plt.figure(figsize=(12, 6))

# 定义三类时段对应的柱子颜色，用于可视化区分
color_normal = '#4C72B0'
color_before = '#DD8452'
color_late = '#55A868'

# 遍历0-23小时，为每个小时生成对应的颜色值，存入列表供柱状图使用
bar_colors = []
for h in range(24):
    if h < 7:
        bar_colors.append(color_before)
    elif h >= 22:
        bar_colors.append(color_late)
    else:
        bar_colors.append(color_normal)

# 调用matplotlib原生接口绘制柱状图，x轴为小时，y轴为对应刷卡量
plt.bar(x=range(24), height=hour_counts.values, color=bar_colors, width=0.75)

# 设置x轴刻度，步长为2，调整字体大小
plt.xticks(range(0, 24, 2), fontsize=10)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Swiping Volume (times)', fontsize=12)

# 设置图表英文标题，pad调整标题与图表的间距
plt.title('24-Hour Bus Card Swiping Volume Distribution', fontsize=14, pad=15)

# 自定义图例元素，对应三类时段的颜色与说明
legend_items = [
    Patch(facecolor=color_normal, label='Normal hours (7:00-21:59)'),
    Patch(facecolor=color_before, label='Before morning peak (<7:00)'),
    Patch(facecolor=color_late, label='Late night (≥22:00)')
]
plt.legend(handles=legend_items, loc='upper right', fontsize=9)

# 添加水平方向网格线，设置虚线样式与透明度，提升可读性
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 自动调整布局，避免标签被截断
plt.tight_layout()
# 保存图片到当前目录，设置分辨率为150dpi
plt.savefig('hour_distribution.png', dpi=150)
plt.close()
print("\n(b) 已保存24小时分布柱状图: hour_distribution.png")
