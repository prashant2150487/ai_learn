import numbers
from unittest import result
# from os import event

import numpy as np
from setuptools import sic

# arr=np.array([1,2,3,4])
# print(arr)
# print(arr+10)
# matrix = np.array([[1,2,3],[4,5,6]])
# vector = np.array([1,2,0])
# print(vector)
# print(matrix)
# print(vector+matrix)

# arr = np.array([[1,2,3],[4,5,6]])

# print(arr)
# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))
# print(np.min(arr))

# print(np.sum(arr,axis=1))
# print(np.sum(arr,axis=0))


# arr=np.array([1,2,3,4])
# print(arr)
# even= arr[arr % 2==0]
# print(even)


# arr=np.array([1,2,3,4,5,6,7])
# evens=arr[arr%2]
# print("evens",evens)

# arr[arr>3]=0
# print(arr)

# gernrate reandom numbers

# random_array=np.random.rand(3,3)
# print(random_array)

# random_array1=np.random.randint(0,10,size=(2,3))
# print(random_array1)

# ---------------------------

# matrix=np.array([[1,2,3],[4,5,6],[7,8,9]])
# # print(matrix)

# vector=np.array([1,-1,5])
# print(vector)

# result=vector+matrix
# print(result)

dataset=np.random.randint(1,51,size=(5,5))

print(dataset)
# dataset[dataset>50]=0
# print(dataset)

print(np.sum(dataset))
print(np.mean(dataset))
print(np.std(dataset))







