import numpy as np
import matplotlib.pyplot as plt

# Step Factor Initializations
rho = 0.25
eta = 1
lamda = 1
mu = 2.5
epsilon = 10**(-5)

# Define phi as arrays
phi1 = np.zeros((400, 1))
phi2 = np.zeros((400, 1))
phi3 = np.zeros((400, 1))
phi4 = np.zeros((400, 1))

# Input initializations
u1 = np.zeros((400, 1))
u2 = np.zeros((400, 1))
u3 = np.zeros((400, 1))
u4 = np.zeros((400, 1))

#Error initializations
e1 = np.zeros((401, 1))
e2 = np.zeros((401, 1))
e3 = np.zeros((401, 1))
e4 = np.zeros((401, 1))

# Definition of yd, si and y as arrays
y1 = np.zeros((401, 1))
y2 = np.zeros((401, 1))
y3 = np.zeros((401, 1))
y4 = np.zeros((401, 1))

si1 = np.zeros((400, 1))
si2 = np.zeros((400, 1))
si3 = np.zeros((400, 1))
si4 = np.zeros((400, 1))
yd = np.zeros((401 ,1))

for i in range (400):
    yd[i] = 0.5*np.sin(k*np.pi/30)+0.3*np.cos(k*np.pi/10);


for k in range(400):
    for i in range(400):
        #Estimator
        if k == 1:
            phi1[1] = 2
            phi2[1] = 2
            phi3[1] = 2
            phi4[1] = 2
        elif k == 2:
            phi1[k,i] = phi1[k-1] + (eta * (u1[k-1]-0) / (mu + (u1[k-1]-0)**2)) * (y1[k] - y1[k-1] - phi1[k-1] * (u1[k-1]-0))
            phi2[k,i] = phi2[k-1] + (eta * (u2[k-1]-0) / (mu + (u2[k-1]-0)**2)) * (y2[k] - y2[k-1] - phi2[k-1] * (u2[k-1]-0))
            phi3[k,i] = phi3[k-1] + (eta * (u3[k-1]-0) / (mu + (u3[k-1]-0)**2)) * (y3[k] - y3[k-1] - phi3[k-1] * (u3[k-1]-0))
            phi4[k,i] = phi4[k-1] + (eta * (u4[k-1]-0) / (mu + (u4[k-1]-0)**2)) * (y4[k] - y4[k-1] - phi4[k-1] * (u4[k-1]-0))
        else:
            phi1[k,i] = phi1[k-1] + (eta * (u1[k-1] - u1[k-2]) / (mu + (u1[k-1] - u1[k-2])**2)) * (y1[k] - y1[k-1] - phi1[k-1] * (u1[k-1] - u1[k-2]))
            phi2[k,i] = phi2[k-1] + (eta * (u2[k-1] - u2[k-2]) / (mu + (u2[k-1] - u2[k-2])**2)) * (y2[k] - y2[k-1] - phi2[k-1] * (u2[k-1] - u2[k-2]))
            phi3[k,i] = phi3[k-1] + (eta * (u3[k-1] - u3[k-2]) / (mu + (u3[k-1] - u3[k-2])**2)) * (y3[k] - y3[k-1] - phi3[k-1] * (u3[k-1] - u3[k-2]))
            phi4[k,i] = phi4[k-1] + (eta * (u4[k-1] - u4[k-2]) / (mu + (u4[k-1] - u4[k-2])**2)) * (y4[k] - y4[k-1] - phi4[k-1] * (u4[k-1] - u4[k-2]))
        
        #Definition of si

        if k == 1:
            si1[1] = 0
            si2[1] = 0
            si3[1] = 0
            si4[1] = 0
        else:
            si1[k,i] = yd[k] - 2*y1[k] + y4[k]
            si2[k,i] = y1[k] - 2*y2[k] + y3[k]
            si3[k,i] = y2[k] + yd[k] - 2*y3[k]
            si4[k,i] = y1[k] + y3[k] - 2*y4[k]

        #input

        if k == 1:
            u1[1] = 0
            u2[1] = 0
            u3[1] = 0
            u4[1] = 0
        else:
            u1[k,i] = u1(k,i-1) + (rho * phi1[k,i] / (lamda + abs(phi1[k,i])**2)) * si1[k+1,i-1]

        # Update y values, adding safeguards for large values
        y1[1] = 0.51
        y2[1] = 2.5
        y3[1] = 3.5
        y4[1] = 4

        

        # y1[k+1,i] = y1[k+1, i-1] + phi1[k,i]

        y1[i, k+1] = y1[i, k] * u1[i, k] / (1 + y1[i, k]**2) + u1[i, k]


        #hetrogenious expression multi agents system
