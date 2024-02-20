import cv2
import numpy as np

a = np.zeros((4, 3, 2))
count = 0
for i in range(4):
    for j in range(3):
        for k in range(2):
            a[i, j, k] = count
            count += 1

b = np.zeros((4, 1, 2))
# b = np.zeros((4, 2))
count = 0
for i in range(4):
    for j in range(1):
        for k in range(2):
            b[i, j, k] = count
            count += 2

print(a[:2])
print("-----")
print(b)
print("-----")

c = a - b
print(c[:2])

print("-----")

t = np.random.randint(1, 3, (4, 3)).reshape(4, 3, 1)
print(t[:2])
print("-----")
d = a * t
print(d[:2])