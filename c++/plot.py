import numpy as np
import matplotlib.pyplot as plt
import sys,os

cuda_cost = []
cuda_f = open("cuda.txt", "r", encoding="UTF-8")
cuda_lines = cuda_f.readlines()
for line in cuda_lines:
    cuda_cost.append(float(line))

cpu_cost = []
cpu_f = open("cpu.txt", "r", encoding="UTF-8")
cpu_lines = cpu_f.readlines()
for line in cpu_lines:
    cpu_cost.append(float(line))

x_size = min(len(cpu_cost), len(cuda_cost))
print(x_size)
x = range(0,x_size)
# y1
plt.plot(x, cuda_cost)

# y2
plt.plot(x, cpu_cost)

# 显示图像
plt.show()