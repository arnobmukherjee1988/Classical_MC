import numpy as np
import numba

# Compute magnetization
@numba.njit
def compute_magnetization(sx, sy, sz, area):
    mx = 0.0
    my = 0.0
    mz = 0.0
    
    for i in range(area):
        mx = mx + sx[i]
        my = my + sy[i]
        mz = mz + sz[i]
    
    mx = mx / float(area)
    my = my / float(area)
    mz = mz / float(area)
    mag = np.sqrt(mx**2 + my**2 + mz**2)
    
    return mx, my, mz, mag

# Compute spin structure factor
def compute_structure_factor_fft(sx, sy, sz, nx, ny):    
    # Reshape spins to 2D lattice
    sx_2d = sx.reshape(ny, nx)
    sy_2d = sy.reshape(ny, nx)
    sz_2d = sz.reshape(ny, nx)
    
    # Compute 2D FFT for each component
    sfx = np.fft.fft2(sx_2d)
    sfy = np.fft.fft2(sy_2d)
    sfz = np.fft.fft2(sz_2d)
    
    # Normalize
    sfx = sfx / float(nx * ny)
    sfy = sfy / float(nx * ny)
    sfz = sfz / float(nx * ny)
    
    # Compute power spectrum
    sf_x = np.abs(sfx)**2
    sf_y = np.abs(sfy)**2
    sf_z = np.abs(sfz)**2
    sf = sf_x + sf_y + sf_z
    
    # Shift zero frequency to center for output
    sf_x = np.fft.fftshift(sf_x)
    sf_y = np.fft.fftshift(sf_y)
    sf_z = np.fft.fftshift(sf_z)
    sf = np.fft.fftshift(sf)
    
    # match original output size (nx+1, ny+1)
    sf_x_padded = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    sf_y_padded = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    sf_z_padded = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    sf_padded = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    
    sf_x_padded[:nx, :ny] = sf_x
    sf_y_padded[:nx, :ny] = sf_y
    sf_z_padded[:nx, :ny] = sf_z
    sf_padded[:nx, :ny] = sf
    
    return sf_x_padded, sf_y_padded, sf_z_padded, sf_padded

# Compute structure factor of tilde spin
def compute_structure_factor_tildemag_fft(sx_t, sy_t, sz_t, nx, ny):
    # Compute magnitude
    mag_t = np.sqrt(sx_t**2 + sy_t**2 + sz_t**2)
    
    # Reshape to 2D
    mag_t_2d = mag_t.reshape(ny, nx)
    
    # Compute FFT
    sf_mag = np.fft.fft2(mag_t_2d)
    sf_mag = sf_mag / float(nx * ny)
    
    # Power spectrum
    sf_tildemag = np.abs(sf_mag)**2
    
    # Shift and pad
    sf_tildemag = np.fft.fftshift(sf_tildemag)
    sf_tildemag_padded = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    sf_tildemag_padded[:nx, :ny] = sf_tildemag
    
    return sf_tildemag_padded

# Compute local density of skyrmions
@numba.njit
def compute_skyrmion_density(sx, sy, sz, nx, ny):
    Pi = np.arccos(-1.0)
    chi_dis = np.zeros((nx, ny), dtype=np.float64)
    
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            i = (iy - 1) * nx + ix - 1
            
            # Positive x neighbour
            jy = iy
            jx = ix + 1
            if jx > nx:
                jx = 1
            j1 = (jy - 1) * nx + jx - 1
            
            # Positive y neighbour
            jx = ix
            jy = iy + 1
            if jy > ny:
                jy = 1
            j2 = (jy - 1) * nx + jx - 1
            
            # Negative x neighbour
            jy = iy
            jx = ix - 1
            if jx < 1:
                jx = nx
            j3 = (jy - 1) * nx + jx - 1
            
            # Negative y neighbour
            jx = ix
            jy = iy - 1
            if jy < 1:
                jy = ny
            j4 = (jy - 1) * nx + jx - 1
            
            chi1 = ((1.0 / (8.0 * Pi)) * 
                   (sx[i] * (sy[j1] * sz[j2] - sz[j1] * sy[j2]) +
                    sy[i] * (sz[j1] * sx[j2] - sx[j1] * sz[j2]) +
                    sz[i] * (sx[j1] * sy[j2] - sy[j1] * sx[j2])))
            
            chi2 = ((1.0 / (8.0 * Pi)) * 
                   (sx[i] * (sy[j3] * sz[j4] - sz[j3] * sy[j4]) +
                    sy[i] * (sz[j3] * sx[j4] - sx[j3] * sz[j4]) +
                    sz[i] * (sx[j3] * sy[j4] - sy[j3] * sx[j4])))
            
            chi_dis[ix - 1, iy - 1] = chi1 + chi2
    
    return chi_dis


# Compute helicity
@numba.njit
def compute_helicity(sx, sy, sz, nx, ny, area):
    helicity_j1_x = 0.0
    helicity_j1_y = 0.0
    helicity_j1_z = 0.0
    helicity_j2_x = 0.0
    helicity_j2_y = 0.0
    helicity_j2_z = 0.0
    
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            i = (iy - 1) * nx + ix - 1
            
            # Positive x neighbour
            jy = iy
            jx = ix + 1
            if jx > nx:
                jx = 1
            j1 = (jy - 1) * nx + jx - 1
            
            # Positive y neighbour
            jx = ix
            jy = iy + 1
            if jy > ny:
                jy = 1
            j2 = (jy - 1) * nx + jx - 1
            
            helicity_j1_x = helicity_j1_x + (sy[i] * sz[j1] - sz[i] * sy[j1]) / float(area)
            helicity_j1_y = helicity_j1_y - (sx[i] * sz[j1] - sz[i] * sx[j1]) / float(area)
            helicity_j1_z = helicity_j1_z + (sx[i] * sy[j1] - sy[i] * sx[j1]) / float(area)
            helicity_j2_x = helicity_j2_x + (sy[i] * sz[j2] - sz[i] * sy[j2]) / float(area)
            helicity_j2_y = helicity_j2_y - (sx[i] * sz[j2] - sz[i] * sx[j2]) / float(area)
            helicity_j2_z = helicity_j2_z + (sx[i] * sy[j2] - sy[i] * sx[j2]) / float(area)
    
    return helicity_j1_x, helicity_j1_y, helicity_j1_z, helicity_j2_x, helicity_j2_y, helicity_j2_z

# Compute spin chirality on every square plaquette
@numba.njit
def compute_chirality(sx, sy, sz, cx, cy, nx, ny):
    chiral = np.zeros((nx, ny), dtype=np.float64)
    sx_c = np.zeros((cx, cy), dtype=np.float64)
    sy_c = np.zeros((cx, cy), dtype=np.float64)
    sz_c = np.zeros((cx, cy), dtype=np.float64)
    
    for iy in range(1, ny + 1):
        for ix in range(1, nx + 1):
            # Construct square plaquette for this site
            for iy_c in range(1, cy + 1):
                for ix_c in range(1, cx + 1):
                    if cx % 2 != 0:
                        ix_l = ix_c - (cx + 1) // 2
                        iy_l = iy_c - (cy + 1) // 2
                    else:
                        ix_l = ix_c - cx // 2
                        iy_l = iy_c - cy // 2
                    
                    ix_p = ix + ix_l
                    iy_p = iy + iy_l
                    
                    if ix_p > nx:
                        ix_p = ix_p - nx
                    if ix_p <= 0:
                        ix_p = nx + ix_p
                    if iy_p > ny:
                        iy_p = iy_p - ny
                    if iy_p <= 0:
                        iy_p = ny + iy_p
                    
                    i_p = (iy_p - 1) * nx + ix_p - 1
                    sx_c[ix_c - 1, iy_c - 1] = sx[i_p]
                    sy_c[ix_c - 1, iy_c - 1] = sy[i_p]
                    sz_c[ix_c - 1, iy_c - 1] = sz[i_p]
            
            # Calculate chirality for plaquette
            chiral_triangle1 = (sx_c[0, 0] * (sy_c[1, 0] * sz_c[1, 1] - sz_c[1, 0] * sy_c[1, 1]) -
                               sy_c[0, 0] * (sx_c[1, 0] * sz_c[1, 1] - sz_c[1, 0] * sx_c[1, 1]) +
                               sz_c[0, 0] * (sx_c[1, 0] * sy_c[1, 1] - sy_c[1, 0] * sx_c[1, 1]))
            
            chiral_triangle2 = (sx_c[1, 0] * (sy_c[1, 1] * sz_c[0, 1] - sz_c[1, 1] * sy_c[0, 1]) -
                               sy_c[1, 0] * (sx_c[1, 1] * sz_c[0, 1] - sz_c[1, 1] * sx_c[0, 1]) +
                               sz_c[1, 0] * (sx_c[1, 1] * sy_c[0, 1] - sy_c[1, 1] * sx_c[0, 1]))
            
            chiral_triangle3 = (sx_c[1, 1] * (sy_c[0, 1] * sz_c[0, 0] - sz_c[0, 1] * sy_c[0, 0]) -
                               sy_c[1, 1] * (sx_c[0, 1] * sz_c[0, 0] - sz_c[0, 1] * sx_c[0, 0]) +
                               sz_c[1, 1] * (sx_c[0, 1] * sy_c[0, 0] - sy_c[0, 1] * sx_c[0, 0]))
            
            chiral_triangle4 = (sx_c[0, 1] * (sy_c[0, 0] * sz_c[1, 0] - sz_c[0, 0] * sy_c[1, 0]) -
                               sy_c[0, 1] * (sx_c[0, 0] * sz_c[1, 0] - sz_c[0, 0] * sx_c[1, 0]) +
                               sz_c[0, 1] * (sx_c[0, 0] * sy_c[1, 0] - sy_c[0, 0] * sx_c[1, 0]))
            
            chiral[ix - 1, iy - 1] = (chiral_triangle1 + chiral_triangle2 + chiral_triangle3 + chiral_triangle4) / 4.0
    
    return chiral
