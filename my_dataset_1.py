import random
list1 = [1, 2, 3, 4, 5, 6, 7]  # 所有的数据
batch_size = 2
epoch = 10
shuffle = True

print(list1)
for e in range(epoch):
    if shuffle:
        random.shuffle(list1)  # 无返回值
    for i in range(0, len(list1), batch_size):  # 数据加载的过程
        batch_data = list1[i:i+batch_size]
        print(batch_data)
