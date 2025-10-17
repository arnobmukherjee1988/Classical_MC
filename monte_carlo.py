import numpy as np
import numba

@numba.njit
def initialize_random_spins_numba(area, spin_val, Pi):
    # Initialize random spin configuration
    sx = np.zeros(area, dtype=np.float64)
    sy = np.zeros(area, dtype=np.float64)
    sz = np.zeros(area, dtype=np.float64)
    theta = np.zeros(area, dtype=np.float64)
    phi = np.zeros(area, dtype=np.float64)
    
    for i in range(area):
        r1 = np.random.random()
        phi[i] = float(int(120.0 * r1)) * 2.0 * Pi / 120.0
        
        r2 = np.random.random()
        theta[i] = np.arccos(float(int(60.0 * r2)) / (60.0 - 1.0))
        
        r3 = np.random.random()
        if r3 > 0.5:
            theta[i] = Pi - theta[i]
        
        sx[i] = spin_val * np.sin(theta[i]) * np.cos(phi[i])
        sy[i] = spin_val * np.sin(theta[i]) * np.sin(phi[i])
        sz[i] = spin_val * np.cos(theta[i])
    
    return sx, sy, sz, theta, phi

# Perform one Monte Carlo sweep over all sites
@numba.njit
def monte_carlo_step_numba(sx, sy, sz, sx_t, sy_t, sz_t, theta, phi, nx, ny, tx, ty, vr, 
                           Dx, Dy, Bx_zmn, By_zmn, Bz_zmn, Temp, spin_val, ratio_val, Pi):
        
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            i = (iy - 1) * nx + ix - 1
            
            # Get neighbour indices
            jy = iy
            jx = ix + 1
            if jx > nx:
                jx = 1
            j1 = (jy - 1) * nx + jx - 1
            
            jx = ix
            jy = iy + 1
            if jy > ny:
                jy = 1
            j2 = (jy - 1) * nx + jx - 1
            
            jy = iy
            jx = ix - 1
            if jx < 1:
                jx = nx
            j3 = (jy - 1) * nx + jx - 1
            
            jx = ix
            jy = iy - 1
            if jy < 1:
                jy = ny
            j4 = (jy - 1) * nx + jx - 1
            
            # Compute old energy
            s_mod_x1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j1]**2 + sy_t[j1]**2 + sz_t[j1]**2)
            s_mod_x2 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j3]**2 + sy_t[j3]**2 + sz_t[j3]**2)
            s_mod_y1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j2]**2 + sy_t[j2]**2 + sz_t[j2]**2)
            s_mod_y2 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j4]**2 + sy_t[j4]**2 + sz_t[j4]**2)
            
            f_term_x1_old = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j1] + sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) +
                            (vr**2) * (1.0 - (sx_t[i]*sx_t[j1] - sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) -
                            (2.0*tx*vr) * ((sx_t[i]*sz_t[j1] - sz_t[i]*sx_t[j1])/s_mod_x1))
            
            f_term_x2_old = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j3] + sy_t[i]*sy_t[j3] + sz_t[i]*sz_t[j3])/s_mod_x2) +
                            (vr**2) * (1.0 - (sx_t[i]*sx_t[j3] - sy_t[i]*sy_t[j3] + sz_t[i]*sz_t[j3])/s_mod_x2) -
                            (2.0*tx*vr) * ((sx_t[j3]*sz_t[i] - sz_t[j3]*sx_t[i])/s_mod_x2))
            
            if f_term_x1_old < 0.0:
                f_term_x1_old = 0.0
            if f_term_x2_old < 0.0:
                f_term_x2_old = 0.0
            
            f_x_old = -Dx * np.sqrt(f_term_x1_old / 2.0) - Dx * np.sqrt(f_term_x2_old / 2.0)
            
            f_term_y1_old = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j2] + sy_t[i]*sy_t[j2] + sz_t[i]*sz_t[j2])/s_mod_y1) +
                            (vr**2) * (1.0 + (sx_t[i]*sx_t[j2] - sy_t[i]*sy_t[j2] - sz_t[i]*sz_t[j2])/s_mod_y1) +
                            (2.0*ty*vr) * ((sz_t[i]*sy_t[j2] - sy_t[i]*sz_t[j2])/s_mod_y1))
            
            f_term_y2_old = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j4] + sy_t[i]*sy_t[j4] + sz_t[i]*sz_t[j4])/s_mod_y2) +
                            (vr**2) * (1.0 + (sx_t[i]*sx_t[j4] - sy_t[i]*sy_t[j4] - sz_t[i]*sz_t[j4])/s_mod_y2) +
                            (2.0*ty*vr) * ((sz_t[j4]*sy_t[i] - sy_t[j4]*sz_t[i])/s_mod_y2))
            
            if f_term_y1_old < 0.0:
                f_term_y1_old = 0.0
            if f_term_y2_old < 0.0:
                f_term_y2_old = 0.0
            
            f_y_old = -Dy * np.sqrt(f_term_y1_old / 2.0) - Dy * np.sqrt(f_term_y2_old / 2.0)
            
            Eng_B_zmn = -Bx_zmn * sx_t[i] - By_zmn * sy_t[i] - Bz_zmn * sz_t[i]
            effE_old = f_x_old + f_y_old + Eng_B_zmn
            
            # Save old configuration
            theta_temp = theta[i]
            phi_temp = phi[i]
            
            # Generate new random spin
            r1 = np.random.random()
            phi[i] = float(int(120.0 * r1)) * 2.0 * Pi / 120.0
            
            r2 = np.random.random()
            theta[i] = np.arccos(float(int(60.0 * r2)) / (60.0 - 1.0))
            
            r3 = np.random.random()
            if r3 > 0.5:
                theta[i] = Pi - theta[i]
            
            sx[i] = spin_val * np.sin(theta[i]) * np.cos(phi[i])
            sy[i] = spin_val * np.sin(theta[i]) * np.sin(phi[i])
            sz[i] = spin_val * np.cos(theta[i])
            
            sx_t[i] = sx[i] - ratio_val * sy[i]
            sy_t[i] = sy[i] + ratio_val * sx[i]
            sz_t[i] = sz[i]
            
            # Compute new energy
            s_mod_x1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j1]**2 + sy_t[j1]**2 + sz_t[j1]**2)
            s_mod_x2 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j3]**2 + sy_t[j3]**2 + sz_t[j3]**2)
            s_mod_y1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j2]**2 + sy_t[j2]**2 + sz_t[j2]**2)
            s_mod_y2 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j4]**2 + sy_t[j4]**2 + sz_t[j4]**2)
            
            f_term_x1_new = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j1] + sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) +
                            (vr**2) * (1.0 - (sx_t[i]*sx_t[j1] - sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) -
                            (2.0*tx*vr) * ((sx_t[i]*sz_t[j1] - sz_t[i]*sx_t[j1])/s_mod_x1))
            
            f_term_x2_new = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j3] + sy_t[i]*sy_t[j3] + sz_t[i]*sz_t[j3])/s_mod_x2) +
                            (vr**2) * (1.0 - (sx_t[i]*sx_t[j3] - sy_t[i]*sy_t[j3] + sz_t[i]*sz_t[j3])/s_mod_x2) -
                            (2.0*tx*vr) * ((sx_t[j3]*sz_t[i] - sz_t[j3]*sx_t[i])/s_mod_x2))
            
            if f_term_x1_new < 0.0:
                f_term_x1_new = 0.0
            if f_term_x2_new < 0.0:
                f_term_x2_new = 0.0
            
            f_x_new = -Dx * np.sqrt(f_term_x1_new / 2.0) - Dx * np.sqrt(f_term_x2_new / 2.0)
            
            f_term_y1_new = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j2] + sy_t[i]*sy_t[j2] + sz_t[i]*sz_t[j2])/s_mod_y1) +
                            (vr**2) * (1.0 + (sx_t[i]*sx_t[j2] - sy_t[i]*sy_t[j2] - sz_t[i]*sz_t[j2])/s_mod_y1) +
                            (2.0*ty*vr) * ((sz_t[i]*sy_t[j2] - sy_t[i]*sz_t[j2])/s_mod_y1))
            
            f_term_y2_new = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j4] + sy_t[i]*sy_t[j4] + sz_t[i]*sz_t[j4])/s_mod_y2) +
                            (vr**2) * (1.0 + (sx_t[i]*sx_t[j4] - sy_t[i]*sy_t[j4] - sz_t[i]*sz_t[j4])/s_mod_y2) +
                            (2.0*ty*vr) * ((sz_t[j4]*sy_t[i] - sy_t[j4]*sz_t[i])/s_mod_y2))
            
            if f_term_y1_new < 0.0:
                f_term_y1_new = 0.0
            if f_term_y2_new < 0.0:
                f_term_y2_new = 0.0
            
            f_y_new = -Dy * np.sqrt(f_term_y1_new / 2.0) - Dy * np.sqrt(f_term_y2_new / 2.0)
            
            Eng_B_zmn = -Bx_zmn * sx_t[i] - By_zmn * sy_t[i] - Bz_zmn * sz_t[i]
            effE_new = f_x_new + f_y_new + Eng_B_zmn
            
            # Metropolis algorithm
            diffE = effE_new - effE_old
            seed = np.random.random()
            
            if np.abs(diffE / Temp) < 8.0:
                prob = np.exp(-diffE / Temp)
            else:
                prob = 0.0
            
            accept = False
            if diffE <= 0.0:
                accept = True
            elif prob > seed:
                accept = True
            
            if not accept:
                theta[i] = theta_temp
                phi[i] = phi_temp
                sx[i] = spin_val * np.sin(theta[i]) * np.cos(phi[i])
                sy[i] = spin_val * np.sin(theta[i]) * np.sin(phi[i])
                sz[i] = spin_val * np.cos(theta[i])
                sx_t[i] = sx[i] - ratio_val * sy[i]
                sy_t[i] = sy[i] + ratio_val * sx[i]
                sz_t[i] = sz[i]
    
    return sx, sy, sz, sx_t, sy_t, sz_t, theta, phi