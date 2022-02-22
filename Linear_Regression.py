# coding: utf-8
# 线性回归：房价预测(f = k*x + b) f(房价), x(年份)为已知；k, b 为未知
import numpy as np
years = np.array([i for i in range(2000, 2022)])  # 年份 2000～2021
prices = np.array([10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000,
                   23000, 26000, 35000, 30000, 40000, 45000, 52000, 50000, 60000])
epoch = 10000
k = 1
b = 1
lr = 0.0000001
for e in range(epoch):
    for x, label in zip(years, prices):
        pre = k * x + b
        loss = (pre - label) ** 2

        delta_k = 2 * (pre - label) * x
        delta_b = 2 * (pre - label)

        k = k - delta_k * lr
        b = b - delta_b * lr
print(f"k = {k}, b = {b}")
while True:
    year = float(input("请输入年份："))
    print("预测的房价为：", end=" ")
    print(int(year) * k + b)

