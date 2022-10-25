import numpy as np
import torch

arr1 = np.arange(24).reshape((3,2,4))
print(arr1)
print(arr1.transpose((1,0,2)))

arr2 = np.arange(4).reshape((2,2))
print(arr2.transpose((1,0)))
# 2维以上的transpose没有行列的概念，有的只是维度的概念，transpose就是调换元素维度来调换元素位置
# (1,0,2)第0维的下标和第1维的下标对换，第2维不变。换的是元素的位置而不是元素，倒不如说用旧元素、新位置来创建了一个新数组。

input=torch.arange(0,6)
# print(input)
# print(input.shape)
# print(input.unsqueeze(0))
# print(input.unsqueeze(0).shape)
# print(input.unsqueeze(1))
# print(input.unsqueeze(1).shape)

arr1 = np.arange(4)
def sq(x):
    return x*x

print(list(map(sq,arr1)))