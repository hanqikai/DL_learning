# coding : utf-8
# (x - 2)**2 = 0
# loss对x的导数的几何意义：指导x下一次变化之后的大小和方向
# 初始值x对于模型非常重要，初始值设置的过大，会导致梯度爆炸（为什么？）
# 常用的初始化方法：凯明初始化等
from tqdm import trange
epoch = 1000
lr = 0.05
label = 0
x = 50
for e in trange(epoch):
    pre = (x - 2) ** 2
    loss = (pre - label) ** 2
    delta_x = 2 * (pre - label) * 2 * (x - 2)
    x -= delta_x * lr
print(x)
