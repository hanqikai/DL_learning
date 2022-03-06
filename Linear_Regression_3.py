# 两个特征
import numpy as np


class MyDataset:
    def __init__(self, xs, ys, zs, batch_size, shuffle):
        self.xs = xs
        self.ys = ys
        self.zs = zs
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
        z = self.dataset.zs[index]
        self.cursor += self.dataset.batch_size
        return x, y, z


if __name__ == '__main__':
    years = np.array([i for i in range(2000, 2022)])  # 年份 2000～2021
    years = (years - 2000) / 22  # batch_normalizer、 layer_normalizer
    floors = np.array([i for i in range(23, 1, -1)]) / 23
    prices = np.array([10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000, 20000, 19000, 22000, 24000,
                       23000, 26000, 35000, 30000, 40000, 45000, 52000, 50000, 60000]) / 60000
    # 数据归一化：除以最大值，z-score归一化，min-max
    # 线性回归：房价预测(f = k*x + b) f(房价), x(年份)为已知；k, b 为未知
    epoch = 10000
    k_1 = 1
    k_2 = -1
    b = 0
    lr = 0.07
    batch_size = 2
    shuffle = True
    loss = 0
    dataset = MyDataset(years, prices, floors, batch_size, shuffle)
    for e in range(epoch):
        for year, price, floor in dataset:
            pre = k_1 * year + k_2 * floor + b
            loss = np.sum((pre - price) ** 2)

            delta_k_1 = np.sum(2 * (pre - price) * year)
            delta_k_2 = np.sum(2 * (pre - price) * floor)
            delta_b = np.sum(2 * (pre - price))

            k_1 -= delta_k_1 * lr
            k_2 -= delta_k_2 * lr
            b -= delta_b * lr
        if e % 100 == 0:
            print(loss)
    while True:
        year = (int(input("请输入年份：")) - 2000) / 22
        floor = int(input("请输入楼层：")) / 23
        print("预测的房价为：", end=" ")
        print((year * k_1 + floor * k_2 + b) * 60000)
# 数据处理
# 参数初始化
# 构建数据集
# 模型推理
# 梯度计算（反向传播）
# 参数更新
# 模型预测
