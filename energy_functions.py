import numpy as np
import numba

@numba.njit
def compute_energy_terms(sx_t, sy_t, sz_t, i, j1, j2, j3, j4, tx, ty, vr, Dx, Dy, Bx_zmn, By_zmn, Bz_zmn):
    # Compute energy terms for a given site and its neighbours
    
    # Compute spin magnitudes
    s_mod_x1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j1]**2 + sy_t[j1]**2 + sz_t[j1]**2)
    s_mod_x2 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j3]**2 + sy_t[j3]**2 + sz_t[j3]**2)
    s_mod_y1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j2]**2 + sy_t[j2]**2 + sz_t[j2]**2)
    s_mod_y2 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j4]**2 + sy_t[j4]**2 + sz_t[j4]**2)
    
    # X-direction terms
    f_term_x1 = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j1] + sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) +
                 (vr**2) * (1.0 - (sx_t[i]*sx_t[j1] - sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) -
                 (2.0*tx*vr) * ((sx_t[i]*sz_t[j1] - sz_t[i]*sx_t[j1])/s_mod_x1))
    
    f_term_x2 = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j3] + sy_t[i]*sy_t[j3] + sz_t[i]*sz_t[j3])/s_mod_x2) +
                 (vr**2) * (1.0 - (sx_t[i]*sx_t[j3] - sy_t[i]*sy_t[j3] + sz_t[i]*sz_t[j3])/s_mod_x2) -
                 (2.0*tx*vr) * ((sx_t[j3]*sz_t[i] - sz_t[j3]*sx_t[i])/s_mod_x2))
    
    if f_term_x1 < 0.0:
        f_term_x1 = 0.0
    if f_term_x2 < 0.0:
        f_term_x2 = 0.0
    
    f_x = -Dx * np.sqrt(f_term_x1 / 2.0) - Dx * np.sqrt(f_term_x2 / 2.0)
    
    # Y-direction terms
    f_term_y1 = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j2] + sy_t[i]*sy_t[j2] + sz_t[i]*sz_t[j2])/s_mod_y1) +
                 (vr**2) * (1.0 + (sx_t[i]*sx_t[j2] - sy_t[i]*sy_t[j2] - sz_t[i]*sz_t[j2])/s_mod_y1) +
                 (2.0*ty*vr) * ((sz_t[i]*sy_t[j2] - sy_t[i]*sz_t[j2])/s_mod_y1))
    
    f_term_y2 = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j4] + sy_t[i]*sy_t[j4] + sz_t[i]*sz_t[j4])/s_mod_y2) +
                 (vr**2) * (1.0 + (sx_t[i]*sx_t[j4] - sy_t[i]*sy_t[j4] - sz_t[i]*sz_t[j4])/s_mod_y2) +
                 (2.0*ty*vr) * ((sz_t[j4]*sy_t[i] - sy_t[j4]*sz_t[i])/s_mod_y2))
    
    if f_term_y1 < 0.0:
        f_term_y1 = 0.0
    if f_term_y2 < 0.0:
        f_term_y2 = 0.0
    
    f_y = -Dy * np.sqrt(f_term_y1 / 2.0) - Dy * np.sqrt(f_term_y2 / 2.0)
    
    # Zeeman energy
    Eng_B_zmn = -Bx_zmn * sx_t[i] - By_zmn * sy_t[i] - Bz_zmn * sz_t[i]
    
    effE = f_x + f_y + Eng_B_zmn
    
    return effE


@numba.njit
def compute_total_energy(sx, sy, sz, sx_t, sy_t, sz_t, nx, ny, tx, ty, vr, Dx, Dy, Bx_zmn, By_zmn, Bz_zmn):
    # Compute total energy of the system
    Eng = 0.0
    Eng_t = 0.0
    
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            i = (iy - 1) * nx + ix - 1
            
            # Neighbour along x axis
            jy = iy
            jx = ix + 1
            if jx > nx:
                jx = 1
            j1 = (jy - 1) * nx + jx - 1
            
            # Neighbour along y axis
            jx = ix
            jy = iy + 1
            if jy > ny:
                jy = 1
            j2 = (jy - 1) * nx + jx - 1
            
            # Compute for real spins
            f_term_x = ((tx**2) * (1.0 + sx[i]*sx[j1] + sy[i]*sy[j1] + sz[i]*sz[j1]) +
                       (vr**2) * (1.0 - sx[i]*sx[j1] + sy[i]*sy[j1] - sz[i]*sz[j1]) -
                       (2.0*tx*vr) * (sx[i]*sz[j1] - sz[i]*sx[j1]))
            
            if f_term_x < 0.0:
                f_term_x = 0.0
            f_x = -Dx * np.sqrt(f_term_x / 2.0)
            
            f_term_y = ((ty**2) * (1.0 + sx[i]*sx[j2] + sy[i]*sy[j2] + sz[i]*sz[j2]) +
                       (vr**2) * (1.0 + sx[i]*sx[j2] - sy[i]*sy[j2] - sz[i]*sz[j2]) +
                       (2.0*ty*vr) * (sz[i]*sy[j2] - sy[i]*sz[j2]))
            
            if f_term_y < 0.0:
                f_term_y = 0.0
            f_y = -Dy * np.sqrt(f_term_y / 2.0)
            
            Eng_B_zmn = -Bx_zmn * sx[i] - By_zmn * sy[i] - Bz_zmn * sz[i]
            Eng = Eng + f_x + f_y + Eng_B_zmn
            
            # Compute for tilde spins
            s_mod_x1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j1]**2 + sy_t[j1]**2 + sz_t[j1]**2)
            s_mod_y1 = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2) * np.sqrt(sx_t[j2]**2 + sy_t[j2]**2 + sz_t[j2]**2)
            
            f_term_x_t = ((tx**2) * (1.0 + (sx_t[i]*sx_t[j1] + sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) +
                         (vr**2) * (1.0 - (sx_t[i]*sx_t[j1] - sy_t[i]*sy_t[j1] + sz_t[i]*sz_t[j1])/s_mod_x1) -
                         (2.0*tx*vr) * ((sx_t[i]*sz_t[j1] - sz_t[i]*sx_t[j1])/s_mod_x1))
            
            if f_term_x_t < 0.0:
                f_term_x_t = 0.0
            f_x_t = -Dx * np.sqrt(f_term_x_t / 2.0)
            
            f_term_y_t = ((ty**2) * (1.0 + (sx_t[i]*sx_t[j2] + sy_t[i]*sy_t[j2] + sz_t[i]*sz_t[j2])/s_mod_y1) +
                         (vr**2) * (1.0 + (sx_t[i]*sx_t[j2] - sy_t[i]*sy_t[j2] - sz_t[i]*sz_t[j2])/s_mod_y1) +
                         (2.0*ty*vr) * ((sz_t[i]*sy_t[j2] - sy_t[i]*sz_t[j2])/s_mod_y1))
            
            if f_term_y_t < 0.0:
                f_term_y_t = 0.0
            f_y_t = -Dy * np.sqrt(f_term_y_t / 2.0)
            
            Eng_B_zmn_t = -Bx_zmn * sx_t[i] - By_zmn * sy_t[i] - Bz_zmn * sz_t[i]
            Eng_t = Eng_t + f_x_t + f_y_t + Eng_B_zmn_t
    
    return Eng, Eng_t