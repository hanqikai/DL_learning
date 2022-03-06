# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


class MyDataset:
    def __init__(self, xs, ys, batch_size, shuffle):
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):  # 返回一个具有__next__的对象
        return DataLoader(self)

    def __len__(self):
        return len(self.xs)


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0
        self.indexs = np.arange(len(self.dataset))
        if self.dataset.shuffle:
            np.random.shuffle(self.indexs)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        index = self.indexs[self.cursor:self.cursor + self.dataset.batch_size]
        x = self.dataset.xs[index]
        y = self.dataset.ys[index]
        self.cursor += self.dataset.batch_size
        return x, y


if __name__ == '__main__':
    years = np.array([i for i in range(2000, 2022)])  # 年份 2000～2021
    years = (years - 2000) / 22  # batch_normalizer、 layer_normalizer
    prices = np.array([10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000,
                       23000, 26000, 35000, 30000, 40000, 45000, 52000, 50000, 60000]) / 60000
    # 数据归一化：除以最大值，z-score归一化，min-max
    # 线性回归：房价预测(f = k*x + b) f(房价), x(年份)为已知；k, b 为未知
    epoch = 10000
    k = 1
    b = 0
    lr = 0.07
    batch_size = 2
    shuffle = True
    dataset = MyDataset(years, prices, batch_size, shuffle)
    for e in range(epoch):
        for year, price in dataset:
            pre = k * year + b
            loss = (pre - price) ** 2

            delta_k = 2 * (pre - price) * year
            delta_b = 2 * (pre - price)

            k -= np.sum(delta_k)/batch_size * lr
            b -= np.sum(delta_b)/batch_size * lr
    while True:
        year = (float(input("请输入年份：")) - 2000) / 22
        print("预测的房价为：", end=" ")
        print((year * k + b) * 60000)
# 数据处理
# 参数初始化
# 构建数据集
# 模型推理
# 梯度计算（反向传播）
# 参数更新
# 模型预测
