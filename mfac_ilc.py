import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = np.array([[1, 0, 0, -1], [-1, 2, -1, 0], [0, -1, 1, 0], [-1, 0, -1, 2]])
D = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

# Step Factor Initializations
rho = 0.3
eta = 1
lamda = 0.5
mu = 0.5
epsilon = 10**(-4)
m = 500  # Number of itterations 
n = 100

# Define phi as arrays
phi1 = np.zeros((m, n))
phi2 = np.zeros((m, n))
phi3 = np.zeros((m, n))
phi4 = np.zeros((m, n))

# Input initializations
u1 = np.zeros((m, n))
u2 = np.zeros((m, n))
u3 = np.zeros((m, n))
u4 = np.zeros((m, n))

# Error initializations
e1 = np.zeros((m, n + 1))
e2 = np.zeros((m, n + 1))
e3 = np.zeros((m, n + 1))
e4 = np.zeros((m, n + 1))

# Definition of yd, si, and y as arrays
y1 = np.zeros((m, n + 1))
y2 = np.zeros((m, n + 1))
y3 = np.zeros((m, n + 1))
y4 = np.zeros((m, n + 1))

si1 = np.zeros((m, n + 1))
si2 = np.zeros((m, n + 1))
si3 = np.zeros((m, n + 1))
si4 = np.zeros((m, n + 1))

# Initialize yd
yd = np.zeros(n + 1)
for k in range(n):
    yd[k] = 0.5 * np.sin(k * np.pi / 30) + 0.3 * np.cos(k * np.pi / 10)

for k in range(1, m):
    for i in range(n):
        # Estimator
        if k == 1:
            phi1[k, i] = 1
            phi2[k, i] = 1
            phi3[k, i] = 1
            phi4[k, i] = 1
        elif k == 2:
            phi1[k, i] = phi1[k-1, i] + (eta * (u1[k-1, i] - 0)) * (y1[k-1, i+1] - 0 - phi1[k-1, i] * (u1[k-1, i] - 0)) / (mu + abs(u1[k-1, i] - 0)**2)
            phi2[k, i] = phi2[k-1, i] + (eta * (u2[k-1, i] - 0)) * (y2[k-1, i+1] - 0 - phi2[k-1, i] * (u2[k-1, i] - 0)) / (mu + abs(u2[k-1, i] - 0)**2)
            phi3[k, i] = phi3[k-1, i] + (eta * (u3[k-1, i] - 0)) * (y3[k-1, i+1] - 0 - phi3[k-1, i] * (u3[k-1, i] - 0)) / (mu + abs(u3[k-1, i] - 0)**2)
            phi4[k, i] = phi4[k-1, i] + (eta * (u4[k-1, i] - 0)) * (y4[k-1, i+1] - 0 - phi4[k-1, i] * (u4[k-1, i] - 0)) / (mu + abs(u4[k-1, i] - 0)**2)
        else:
            phi1[k, i] = phi1[k-1, i] + (eta * (u1[k-1, i] - u1[k-2, i])) * (y1[k-1, i+1] - y1[k-2, i+1] - phi1[k-1, i] * (u1[k-1, i] - u1[k-2, i])) / (mu + abs(u1[k-1, i] - u1[k-2, i])**2)
            phi2[k, i] = phi2[k-1, i] + (eta * (u2[k-1, i] - u2[k-2, i])) * (y2[k-1, i+1] - y2[k-2, i+1] - phi2[k-1, i] * (u2[k-1, i] - u2[k-2, i])) / (mu + abs(u2[k-1, i] - u2[k-2, i])**2)
            phi3[k, i] = phi3[k-1, i] + (eta * (u3[k-1, i] - u3[k-2, i])) * (y3[k-1, i+1] - y3[k-2, i+1] - phi3[k-1, i] * (u3[k-1, i] - u3[k-2, i])) / (mu + abs(u3[k-1, i] - u3[k-2, i])**2)
            phi4[k, i] = phi4[k-1, i] + (eta * (u4[k-1, i] - u4[k-2, i])) * (y4[k-1, i+1] - y4[k-2, i+1] - phi4[k-1, i] * (u4[k-1, i] - u4[k-2, i])) / (mu + abs(u4[k-1, i] - u4[k-2, i])**2)
           
        # Definition of si
        si1[k, i] = yd[i] - 2 * y1[k, i] + y4[k, i]
        si2[k, i] = y1[k, i] - 2 * y2[k, i] + y3[k, i]
        si3[k, i] = y2[k, i] + yd[i] - 2 * y3[k, i]
        si4[k, i] = y1[k, i] + y3[k, i] - 2 * y4[k, i]

        # Input
        if k == 1:
            u1[k, i] = 0
            u2[k, i] = 0
            u3[k, i] = 0
            u4[k, i] = 0
        else:
            u1[k, i] = u1[k-1, i] + rho * phi1[k, i] / (lamda + abs(phi1[k, i])**2) * si1[k-1, i+1]
            u2[k, i] = u2[k-1, i] + rho * phi2[k, i] / (lamda + abs(phi2[k, i])**2) * si2[k-1, i+1]
            u3[k, i] = u3[k-1, i] + rho * phi3[k, i] / (lamda + abs(phi3[k, i])**2) * si3[k-1, i+1]
            u4[k, i] = u4[k-1, i] + rho * phi4[k, i] / (lamda + abs(phi4[k, i])**2) * si4[k-1, i+1]

        # Update y values
        if k == 1:
            y1[k, i+1] = 1.1
            y2[k, i+1] = 1.1
            y3[k, i+1] = 1.1
            y4[k, i+1] = 1.1
        else:
            y1[k, i+1] = y1[k, i]*u1[k,i] / (1 + y1[k, i]**2) + u1[k, i]
            y2[k, i+1] = y2[k, i]*u1[k,i] / (1 + y2[k, i]**2) + u2[k, i]
            y3[k, i+1] = y3[k, i]*u3[k,i] / (1 + y3[k, i]**2) + u3[k, i]
            y4[k, i+1] = y4[k, i]*u4[k,i] / (1 + y4[k, i]**2) + u4[k, i]
        

# Plot results
plt.plot(yd, 'k-', linewidth=1.5, label='$y_d(k)$')
plt.plot(y1[m-1, :], 'r-', marker='o', markersize=4, label='Agent 1')
plt.plot(y2[m-1, :], 'g-', marker='o', markersize=4, label='Agent 2')
plt.plot(y3[m-1, :], 'y-', marker='o', markersize=4, label='Agent 3')
plt.plot(y4[m-1, :], 'b-', marker='o', markersize=4, label='Agent 4')


plt.xlim([0, n])
plt.ylim([-0.8, 0.8])
plt.xlabel('time step')
plt.ylabel('outputs of agents and reference')
plt.legend()
plt.title('Outputs of agents and reference at 200th iteration (Example 1)')
plt.show()
