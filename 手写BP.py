import numpy as np
import struct
import matplotlib.pyplot as plt


def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)


def make_one_hot(labels, class_num=10):
    result = np.zeros((len(labels), class_num))
    for index, lab in enumerate(labels):
        result[index][lab] = 1
    return result


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    return ex/sum_ex


if __name__ == '__main__':
    train_datas = load_images("data/train-images.idx3-ubyte") / 255  # 数据归一化
    train_label = make_one_hot(load_labels("data/train-labels.idx1-ubyte"))

    test_datas = load_images("data/t10k-images.idx3-ubyte")
    test_label = load_labels("data/t10k-labels.idx1-ubyte")
    # print(train_datas.shape, train_label.shape, test_datas.shape, test_label.shape)
    epoch = 20
    batch_size = 6000
    lr = 0.01

    hidden_num = 256
    w1 = np.random.normal(0, 1, size=(784, hidden_num))  # 满足均值为0，方差为1，大小为（784， 256大小的参数矩阵）
    w2 = np.random.normal(0, 1, size=(hidden_num, 10))
    batch_times = int(np.ceil(len(train_datas) / batch_size))  # 训练一轮需要多少次

    for e in range(epoch):
        for batch_index in range(batch_times):
            #  获取数据
            batch_x = train_datas[batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_label = train_label[batch_index * batch_size:(batch_index + 1) * batch_size]
            #  forward处理
            h = batch_x @ w1
            sig_h = sigmoid(h)
            p = sig_h @ w2
            pre = softmax(p)
            # 计算loss
            loss = -np.sum(batch_label * np.log(pre)) / batch_size
            #  前向传播结束
            # backward
            G2 = (pre - batch_label) / batch_size
            delta_w2 = sig_h.T @ G2
            delta_sig_h = G2 @ w2.T
            delta_h = delta_sig_h * sig_h * (1 - sig_h)
            delta_w1 = batch_x.T @ delta_h
            #  更新梯度
            w1 -= lr * delta_w1
            w2 -= lr * delta_w2

        print(loss)






