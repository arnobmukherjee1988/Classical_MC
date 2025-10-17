import numpy as np

# Lattice parameters
nx = 20
ny = 20
area = nx * ny
cx = 2
cy = 2  # for chirality

# Physical constants
Pi = np.arccos(-1.0)
Dx = 1.0
Dy = 1.0
spin = 1.0  # spin magnitude
alpha = 0.3  # parameterization of 't' and 'vr'
t = 1.0 - alpha

# Zeeman magnetic field parameters
no_B_zmn_pt = 10
B_zmn_max = 0.2
B_zmn_min = 0.0
theta_zmn = 0.0
phi_zmn = Pi / 3.0

# Monte Carlo parameters
Neq = 1000  # equilibration steps
Nav = 1000  # averaging steps
Nres = 100
n_confi = 1  # number of configurations

# Improved MC move parameters
# delta_theta = 0.5  # Maximum change in theta per move (tune for ~40% acceptance)
# delta_phi = 0.5    # Maximum change in phi per move (tune for ~40% acceptance)

# Hamiltonian parameters
D = 8.0
JK = 6.0

# Ratio parameters
ratio_min = 0.0
ratio_max = 1.0
ratio_num = 50
ratio = 0.5

# Temperature parameters
max_temp = 0.3
min_temp = 0.001
no_temp = 50

# Create temperature array using linspace
temperature = np.linspace(min_temp, max_temp, no_temp + 1)
num_temp_points = len(temperature)