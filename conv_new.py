import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def get_img(path:str):
    img_files = os.listdir(path)
    result = []
    for file in img_files:
        file = os.path.join(path, file)
        img = cv2.imread(file)
        img = cv2.resize(img, (150, 150))
        img = img.transpose(2, 0, 1)  # 图像转置函数，具体参数的作用还不是很熟悉
        result.append(img)
    return np.array(result)


if __name__ == '__main__':
    imgs = get_img("img")
    print("")
