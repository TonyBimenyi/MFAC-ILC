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