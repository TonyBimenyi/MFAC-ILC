import numpy as np
import matplotlib.pyplot as plt

# Step Factor Initializations
rho = 0.25
eta = 1
lamda = 1
mu = 2.5
epsilon = 10**(-5)
m = 100 #number of iteration

# Define phi as arrays
phi1 = np.zeros((m, 100))
phi2 = np.zeros((m, 100))
phi3 = np.zeros((m, 100))
phi4 = np.zeros((m, 100))

# Input initializations
u1 = np.zeros((m, 100))
u2 = np.zeros((m, 100))
u3 = np.zeros((m, 100))
u4 = np.zeros((m, 100))

# Error initializations
e1 = np.zeros((m, 401))
e2 = np.zeros((m, 401))
e3 = np.zeros((m, 401))
e4 = np.zeros((m, 401))

# Definition of yd, si, and y as arrays
y1 = np.zeros((m, 101))
y2 = np.zeros((m, 101))
y3 = np.zeros((m, 101))
y4 = np.zeros((m, 101))

si1 = np.zeros((m, 101))
si2 = np.zeros((m, 101))
si3 = np.zeros((m, 101))
si4 = np.zeros((m, 101))
yd = np.zeros(101)

for i in range(101):
    yd[i] = 0.5 * np.sin(i * np.pi / 30) + 0.3 * np.cos(i * np.pi / 10)



for k in range(m):
    for i in range(100):
        # Estimator
        if k == 0:
            phi1[k, i] = 2
            phi2[k, i] = 2
            phi3[k, i] = 2
            phi4[k, i] = 2
        elif k == 1:
            phi1[k, i] = phi1[k-1, i] + (eta * (u1[k-1, i] - 0) / (mu + (u1[k-1, i] - 0)**2)) * (y1[k, i] - y1[k-1, i] - phi1[k-1, i] * (u1[k-1, i] - 0))
            phi2[k, i] = phi2[k-1, i] + (eta * (u2[k-1, i] - 0) / (mu + (u2[k-1, i] - 0)**2)) * (y2[k, i] - y2[k-1, i] - phi2[k-1, i] * (u2[k-1, i] - 0))
            phi3[k, i] = phi3[k-1, i] + (eta * (u3[k-1, i] - 0) / (mu + (u3[k-1, i] - 0)**2)) * (y3[k, i] - y3[k-1, i] - phi3[k-1, i] * (u3[k-1, i] - 0))
            phi4[k, i] = phi4[k-1, i] + (eta * (u4[k-1, i] - 0) / (mu + (u4[k-1, i] - 0)**2)) * (y4[k, i] - y4[k-1, i] - phi4[k-1, i] * (u4[k-1, i] - 0))
        else:
            phi1[k, i] = phi1[k-1, i] + (eta * (u1[k-1, i] - u1[k-2, i]) / (mu + abs(u1[k-1, i] - u1[k-2, i])**2)) * (y1[k-1, i+1] - y1[k-2, i+1] - phi1[k-1, i] * (u1[k-1, i] - u1[k-2, i]))
           
        
        # Definition of si
       
        si1 = yd[i] - 2 * y1 + y4
        si2 = y1 - 2 * y2+ y3
        si3 = y2 + yd[i] - 2 * y3
        si4 = y1[i] + y3 - 2 * y4
        
        # Input
        if k == 0:
            u1[k, i] = 0
            u2[k, i] = 0
            u3[k, i] = 0
            u4[k, i] = 0
        else:
            u1[k, i] = u1[k-1, i] + (rho * phi1[k, i] / (lamda + abs(phi1[k, i])**2)) * si1[k-1, i+1]

        
        # Update y values, adding safeguards for large values

        if k== 1:
            y1[k, i+1] = 1
        elif k <= 30:
            y1[k, i+1] = y1[k, i] /  (1 + y1[k, i]**2) + u1[k, i]**2
        else:
            y1[k, i+1] = ((y1[k, i] * y1[k, i-1] * y1[k, i-2] * u1[k, i-1]) + ()) 
         
       

plt.plot(yd, label="Desired Output (yd)")
plt.plot(y1[m-1, :], 'b--', linewidth=1.5, label='y1')
plt.legend()
plt.show()
