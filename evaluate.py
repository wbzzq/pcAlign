import csv
import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'./checkpoints/exp70/errors.csv')

# 去掉第一列，只保留数值列
data_columns = df.iloc[:, 1:]

# 计算每列的均值
mean_values = data_columns.mean()

# 计算每列的方差
variance_values = data_columns.var()

# 计算每列的25分位数
quantile_25_values = data_columns.quantile(0.25)

# 计算每列的75分位数
quantile_75_values = data_columns.quantile(0.75)

# 打印结果并添加标签
print("均值：")
# 获取前三列均值并求和
sum_first_three_means = mean_values.iloc[:3].sum()

# 获取后三列均值并求和
sum_last_three_means = mean_values.iloc[-3:].sum()

print(f"前三列均值的和：{sum_first_three_means}")
print(f"后三列均值的和：{sum_last_three_means}")

print("\n方差：")
for column, var in variance_values.items():
    print(f"{column}: {var}")

print("\n25分位数：")
for column, q25 in quantile_25_values.items():
    print(f"{column}: {q25}")

print("\n75分位数：")
for column, q75 in quantile_75_values.items():
    print(f"{column}: {q75}")
