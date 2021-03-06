import random


class MyDataset:
    def __init__(self, all_datas, batch_size, shuffle=True):
        self.all_data = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0

    # python的魔术方法：某种场景下自动触发的方法
    def __iter__(self):  # 返回一个具有__next__的对象
        if self.shuffle:
            random.shuffle(self.all_data)
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.all_data):
            raise StopIteration
        batch_data = self.all_data[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return batch_data


if __name__ == '__main__':
    all_datas = [1, 2, 3, 4, 5, 6, 7]
    batch_size = 2
    shuffle = True
    epoch = 2

    dataset = MyDataset(all_datas, batch_size, shuffle)
    for e in range(epoch):
        for batch_data in dataset:  # 把一个对象放在for上面时，会自动调用__iter__魔术方法
            print(batch_data)
