import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = np.array([[1, 0, 0, -1], [-1, 2, -1, 0], [0, -1, 1, 0], [-1, 0, -1, 2]])
D = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
m = 100  # Number of iterations
rou = 0.01
lambda_ = 0.5
yita = 1
miu = 0.5

# System output
y1 = np.zeros((m, 101))
y2 = np.zeros((m, 101))
y3 = np.zeros((m, 101))
y4 = np.zeros((m, 101))

# Leader
ydk = np.zeros(101)

# PPD parameters
fai1 = np.zeros((m, 100))
fai2 = np.zeros((m, 100))
fai3 = np.zeros((m, 100))
fai4 = np.zeros((m, 100))

# Controller output
u1 = np.zeros((m, 100))
u2 = np.zeros((m, 100))
u3 = np.zeros((m, 100))
u4 = np.zeros((m, 100))

# Kesai variables
kesai1 = np.zeros((m, 101))
kesai2 = np.zeros((m, 101))
kesai3 = np.zeros((m, 101))
kesai4 = np.zeros((m, 101))

# Leader output
for t in range(101):
    ydk[t] = 0.5 * np.sin(t * np.pi / 30) + 0.3 * np.cos(t * np.pi / 10)

# Iterative algorithm
for k in range(m):
    for t in range(100):
        if k == 1:
            fai1[k, t] = 1
            fai2[k, t] = 1
            fai3[k, t] = 1
            fai4[k, t] = 1
        elif k == 2:
            fai1[k, t] = fai1[k-1, t] + yita * (u1[k-1, t]) * (y1[k-1, t+1] - fai1[k-1, t] * u1[k-1, t]) / (miu + abs(u1[k-1, t])**2)
            fai2[k, t] = fai2[k-1, t] + yita * (u2[k-1, t]) * (y2[k-1, t+1] - fai2[k-1, t] * u2[k-1, t]) / (miu + abs(u2[k-1, t])**2)
            fai3[k, t] = fai3[k-1, t] + yita * (u3[k-1, t]) * (y3[k-1, t+1] - fai3[k-1, t] * u3[k-1, t]) / (miu + abs(u3[k-1, t])**2)
            fai4[k, t] = fai4[k-1, t] + yita * (u4[k-1, t]) * (y4[k-1, t+1] - fai4[k-1, t] * u4[k-1, t]) / (miu + abs(u4[k-1, t])**2)
        else:
            fai1[k, t] = fai1[k-1, t] + yita * (u1[k-1, t] - u1[k-2, t]) * (y1[k-1, t+1] - y1[k-2, t+1] - fai1[k-1, t] * (u1[k-1, t] - u1[k-2, t])) / (miu + abs(u1[k-1, t] - u1[k-2, t])**2)
            fai2[k, t] = fai2[k-1, t] + yita * (u2[k-1, t] - u2[k-2, t]) * (y2[k-1, t+1] - y2[k-2, t+1] - fai2[k-1, t] * (u2[k-1, t] - u2[k-2, t])) / (miu + abs(u2[k-1, t] - u2[k-2, t])**2)
            fai3[k, t] = fai3[k-1, t] + yita * (u3[k-1, t] - u3[k-2, t]) * (y3[k-1, t+1] - y3[k-2, t+1] - fai3[k-1, t] * (u3[k-1, t] - u3[k-2, t])) / (miu + abs(u3[k-1, t] - u3[k-2, t])**2)
            fai4[k, t] = fai4[k-1, t] + yita * (u4[k-1, t] - u4[k-2, t]) * (y4[k-1, t+1] - y4[k-2, t+1] - fai4[k-1, t] * (u4[k-1, t] - u4[k-2, t])) / (miu + abs(u4[k-1, t] - u4[k-2, t])**2)
        
        if k == 1:
            u1[k, t] = 0.01
            u2[k, t] = 0.01
            u3[k, t] = 0.01
            u4[k, t] = 0.01
        else:
            u1[k, t] = u1[k-1, t] + rou * fai1[k, t] * kesai1[k-1, t+1] / (lambda_ + abs(fai1[k, t])**2)
            u2[k, t] = u2[k-1, t] + rou * fai2[k, t] * kesai2[k-1, t+1] / (lambda_ + abs(fai2[k, t])**2)
            u3[k, t] = u3[k-1, t] + rou * fai3[k, t] * kesai3[k-1, t+1] / (lambda_ + abs(fai3[k, t])**2)
            u4[k, t] = u4[k-1, t] + rou * fai4[k, t] * kesai4[k-1, t+1] / (lambda_ + abs(fai4[k, t])**2)
        
        if k == 1:
            y1[k, t+1] = 3.5
            y2[k, t+1] = 3.5
            y3[k, t+1] = 3.5
            y4[k, t+1] = 3.5
        elif t <= 30:
            y1[k, t+1] = y1[k, t] / (1 + y1[k, t]**2) + u1[k, t]**2
            y2[k, t+1] = y2[k, t] / (1 + y2[k, t]**2) + u2[k, t]**2
            y3[k, t+1] = y3[k, t] / (1 + y3[k, t]**2) + u3[k, t]**2
            y4[k, t+1] = y4[k, t] / (1 + y4[k, t]**2) + u4[k, t]**2
        else:
            y1[k, t+1] = (y1[k, t] * y1[k, t-1] * y1[k, t-2] * u1[k, t-1] + (1 + np.random.rand() * (t / 50)) * u1[k, t]) / ((1 + y1[k, t-1]**2) + y1[k, t-2]**2)
            y2[k, t+1] = (y2[k, t] * y2[k, t-1] * y2[k, t-2] * u2[k, t-1] + (1 + np.random.rand() * (t / 50)) * u2[k, t]) / ((1 + y2[k, t-1]**2) + y2[k, t-2]**2)
            y3[k, t+1] = (y3[k, t] * y3[k, t-1] * y3[k, t-2] * u3[k, t-1] + (1 + np.random.rand() * (t / 50)) * u3[k, t]) / ((1 + y3[k, t-1]**2) + y3[k, t-2]**2)
            y4[k, t+1] = (y4[k, t] * y4[k, t-1] * y4[k, t-2] * u4[k, t-1] + (1 + np.random.rand() * (t / 50)) * u4[k, t]) / ((1 + y4[k, t-1]**2) + y4[k, t-2]**2)
        
        kesai1 = y4 + ydk[t] - 2 * y1
        kesai2 = y1 + y3 - 2 * y2
        kesai3 = y2 + ydk[t] - 2 * y3
        kesai4 = y1 + y3 - 2 * y4

# Plotting the results
plt.figure(1)
plt.plot(ydk, 'r-', linewidth=1.5, label='ydk')
plt.plot(y1[m-1, :], 'b--', linewidth=1.5, label='y1')
plt.plot(y2[m-1, :], 'g--', linewidth=1.5, label='y2')
plt.plot(y3[m-1, :], 'c--', linewidth=1.5, label='y3')
plt.plot(y4[m-1, :], 'y--', linewidth=1.5, label='y4')
plt.xlim([1, 100])
plt.legend()
plt.show()

plt.figure(2)
plt.plot(u1[m-1, :], label='u1')
plt.plot(u2[m-1, :], label='u2')
plt.plot(u3[m-1, :], label='u3')
plt.plot(u4[m-1, :], label='u4')
plt.legend()
plt.show()
