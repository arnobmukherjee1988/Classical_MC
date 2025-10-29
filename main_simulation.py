import numpy as np
import os
import time

from input_parameters import *
from energy_functions import compute_total_energy
from observables import (compute_magnetization, compute_structure_factor_fft, 
                               compute_structure_factor_tildemag_fft, compute_skyrmion_density,
                               compute_helicity, compute_chirality)
from monte_carlo import initialize_random_spins_numba, monte_carlo_step_numba
from file_io_functions import (setup_output_directories, open_output_files, close_output_files,
                               write_summary_data, write_all_configuration_files)


def main():
    # Main simulation function
    
    print("Starting Monte Carlo simulation...")
    
    # Set random seed
    np.random.seed(42)
    
    # Get temperature array from input_parameters
    from input_parameters import temperature, num_temp_points
    
    # Arrays to store final spins for each configuration
    sx_final = np.zeros((n_confi, area), dtype=np.float64)
    sy_final = np.zeros((n_confi, area), dtype=np.float64)
    sz_final = np.zeros((n_confi, area), dtype=np.float64)
    theta_final = np.zeros((n_confi, area), dtype=np.float64)
    phi_final = np.zeros((n_confi, area), dtype=np.float64)
    
    # Arrays to store initial spins
    sx_initial = np.zeros((n_confi, area), dtype=np.float64)
    sy_initial = np.zeros((n_confi, area), dtype=np.float64)
    sz_initial = np.zeros((n_confi, area), dtype=np.float64)
    theta_initial = np.zeros((n_confi, area), dtype=np.float64)
    phi_initial = np.zeros((n_confi, area), dtype=np.float64)
    
    # Initialize random spin configurations
    for i_confi in range(n_confi):
        sx_init, sy_init, sz_init, theta_init, phi_init = initialize_random_spins_numba(area, spin, Pi)
        sx_initial[i_confi, :] = sx_init
        sy_initial[i_confi, :] = sy_init
        sz_initial[i_confi, :] = sz_init
        theta_initial[i_confi, :] = theta_init
        phi_initial[i_confi, :] = phi_init
    
    # Loop over Zeeman field values
    for i_B_zmn in range(no_B_zmn_pt + 1):
        B_zmn = B_zmn_min + ((B_zmn_max - B_zmn_min) / float(no_B_zmn_pt)) * float(i_B_zmn)
        
        # Zeeman field components
        Bx_zmn = B_zmn * np.sin(theta_zmn) * np.cos(phi_zmn)
        By_zmn = B_zmn * np.sin(theta_zmn) * np.sin(phi_zmn)
        Bz_zmn = B_zmn * np.cos(theta_zmn)
        
        print(f"\nB_zmn = {B_zmn:.5f} ({i_B_zmn+1}/{no_B_zmn_pt+1})")
        
        # Set hopping parameters
        tx = t
        ty = t
        vr = alpha
        
        # Create output directories
        str_Bzmn = f"{B_zmn:.5f}"
        folder_name = f"./Bzmn_{str_Bzmn}"
        setup_output_directories(folder_name)
        
        # Open all output files
        file_handles = open_output_files(folder_name)
        
        # Temperature loop
        for ncount in range(num_temp_points):
            Temp = temperature[num_temp_points - 1 - ncount]  # decreasing temperature
            t_start = time.time()
            
            # Initialize accumulators
            avE = 0.0
            avE_t = 0.0
            avmx = 0.0
            avmy = 0.0
            avmz = 0.0
            avmag = 0.0
            avmx_t = 0.0
            avmy_t = 0.0
            avmz_t = 0.0
            avmag_t = 0.0
            
            avsf_x = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            avsf_y = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            avsf_z = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            avsf = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            
            avsf_x_t = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            avsf_y_t = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            avsf_z_t = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            avsf_t = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            
            ave_chiral = np.zeros((nx, ny), dtype=np.float64)
            ave_tot_chiral = 0.0
            
            ave_chi_dis = np.zeros((nx, ny), dtype=np.float64)
            ave_sum_chi = 0.0
            ave_chi_dis_t = np.zeros((nx, ny), dtype=np.float64)
            ave_sum_chi_t = 0.0
            
            ave_helicity_j1_x = 0.0
            ave_helicity_j1_y = 0.0
            ave_helicity_j1_z = 0.0
            ave_helicity_j2_x = 0.0
            ave_helicity_j2_y = 0.0
            ave_helicity_j2_z = 0.0
            
            ave_helicity_j1_x_t = 0.0
            ave_helicity_j1_y_t = 0.0
            ave_helicity_j1_z_t = 0.0
            ave_helicity_j2_x_t = 0.0
            ave_helicity_j2_y_t = 0.0
            ave_helicity_j2_z_t = 0.0
            
            avsf_tildemag = np.zeros((nx + 1, ny + 1), dtype=np.float64)
            
            # Configuration loop
            for i_confi in range(n_confi):
                # Initialize or load spins
                sx = np.zeros(area, dtype=np.float64)
                sy = np.zeros(area, dtype=np.float64)
                sz = np.zeros(area, dtype=np.float64)
                sx_t = np.zeros(area, dtype=np.float64)
                sy_t = np.zeros(area, dtype=np.float64)
                sz_t = np.zeros(area, dtype=np.float64)
                theta = np.zeros(area, dtype=np.float64)
                phi = np.zeros(area, dtype=np.float64)
                
                if ncount == 0:
                    for i in range(area):
                        theta[i] = theta_initial[i_confi, i]
                        phi[i] = phi_initial[i_confi, i]
                        sx[i] = sx_initial[i_confi, i]
                        sy[i] = sy_initial[i_confi, i]
                        sz[i] = sz_initial[i_confi, i]
                        sx_t[i] = sx[i] - ratio * sy[i]
                        sy_t[i] = sy[i] + ratio * sx[i]
                        sz_t[i] = sz[i]
                else:
                    for i in range(area):
                        theta[i] = theta_final[i_confi, i]
                        phi[i] = phi_final[i_confi, i]
                        sx[i] = sx_final[i_confi, i]
                        sy[i] = sy_final[i_confi, i]
                        sz[i] = sz_final[i_confi, i]
                        sx_t[i] = sx[i] - ratio * sy[i]
                        sy_t[i] = sy[i] + ratio * sx[i]
                        sz_t[i] = sz[i]
                
                # Monte Carlo iterations
                for iter_val in range(1, Neq + Nav + 1):
                    # Monte Carlo step (JIT compiled - very fast!)
                    sx, sy, sz, sx_t, sy_t, sz_t, theta, phi = monte_carlo_step_numba(
                        sx, sy, sz, sx_t, sy_t, sz_t, theta, phi, nx, ny,
                        tx, ty, vr, Dx, Dy, Bx_zmn, By_zmn, Bz_zmn, Temp, spin, ratio, Pi,
                    )
                    
                    # Save final configuration
                    if iter_val == (Neq + Nav):
                        for i in range(area):
                            theta_final[i_confi, i] = theta[i]
                            phi_final[i_confi, i] = phi[i]
                            sx_final[i_confi, i] = sx[i]
                            sy_final[i_confi, i] = sy[i]
                            sz_final[i_confi, i] = sz[i]
                    
                    # Compute observables after equilibration
                    if iter_val > Neq:
                        # Magnetization (JIT compiled)
                        mx, my, mz, mag = compute_magnetization(sx, sy, sz, area)
                        mx_t, my_t, mz_t, mag_t = compute_magnetization(sx_t, sy_t, sz_t, area)
                        
                        avmx = avmx + mx
                        avmy = avmy + my
                        avmz = avmz + mz
                        avmag = avmag + mag
                        
                        avmx_t = avmx_t + mx_t
                        avmy_t = avmy_t + my_t
                        avmz_t = avmz_t + mz_t
                        avmag_t = avmag_t + mag_t
                        
                        # Energy (JIT compiled)
                        Eng, Eng_t = compute_total_energy(sx, sy, sz, sx_t, sy_t, sz_t, nx, ny,
                                                          tx, ty, vr, Dx, Dy, Bx_zmn, By_zmn, Bz_zmn)
                        avE = avE + Eng / float(area)
                        avE_t = avE_t + Eng_t / float(area)
                        
                        # Skyrmion density (JIT compiled)
                        chi_dis = compute_skyrmion_density(sx, sy, sz, nx, ny)
                        chi_dis_t = compute_skyrmion_density(sx_t, sy_t, sz_t, nx, ny)
                        ave_chi_dis = ave_chi_dis + chi_dis
                        ave_chi_dis_t = ave_chi_dis_t + chi_dis_t
                        
                        # Helicity (JIT compiled)
                        h_j1_x, h_j1_y, h_j1_z, h_j2_x, h_j2_y, h_j2_z = compute_helicity(sx, sy, sz, nx, ny, area)
                        h_j1_x_t, h_j1_y_t, h_j1_z_t, h_j2_x_t, h_j2_y_t, h_j2_z_t = compute_helicity(sx_t, sy_t, sz_t, nx, ny, area)
                        
                        ave_helicity_j1_x = ave_helicity_j1_x + h_j1_x
                        ave_helicity_j1_y = ave_helicity_j1_y + h_j1_y
                        ave_helicity_j1_z = ave_helicity_j1_z + h_j1_z
                        ave_helicity_j2_x = ave_helicity_j2_x + h_j2_x
                        ave_helicity_j2_y = ave_helicity_j2_y + h_j2_y
                        ave_helicity_j2_z = ave_helicity_j2_z + h_j2_z
                        
                        ave_helicity_j1_x_t = ave_helicity_j1_x_t + h_j1_x_t
                        ave_helicity_j1_y_t = ave_helicity_j1_y_t + h_j1_y_t
                        ave_helicity_j1_z_t = ave_helicity_j1_z_t + h_j1_z_t
                        ave_helicity_j2_x_t = ave_helicity_j2_x_t + h_j2_x_t
                        ave_helicity_j2_y_t = ave_helicity_j2_y_t + h_j2_y_t
                        ave_helicity_j2_z_t = ave_helicity_j2_z_t + h_j2_z_t
                        
                        # Time-consuming observables (computed periodically)
                        if (iter_val - Neq) % (Nav // Nres) == 0:
                            # Structure factor (FFT - very fast!)
                            sf_x, sf_y, sf_z, sf = compute_structure_factor_fft(sx, sy, sz, nx, ny)
                            sf_x_t, sf_y_t, sf_z_t, sf_t = compute_structure_factor_fft(sx_t, sy_t, sz_t, nx, ny)
                            
                            avsf_x = avsf_x + sf_x
                            avsf_y = avsf_y + sf_y
                            avsf_z = avsf_z + sf_z
                            avsf = avsf + sf
                            
                            avsf_x_t = avsf_x_t + sf_x_t
                            avsf_y_t = avsf_y_t + sf_y_t
                            avsf_z_t = avsf_z_t + sf_z_t
                            avsf_t = avsf_t + sf_t
                            
                            # Tilde magnitude structure factor (FFT)
                            sf_tildemag = compute_structure_factor_tildemag_fft(sx_t, sy_t, sz_t, nx, ny)
                            avsf_tildemag = avsf_tildemag + sf_tildemag
                            
                            # Chirality (JIT compiled)
                            chiral = compute_chirality(sx, sy, sz, cx, cy, nx, ny)
                            ave_chiral = ave_chiral + chiral
            
            # Average over configurations and Monte Carlo steps
            ave_tot_chiral = np.sum(ave_chiral)
            ave_sum_chi = np.sum(ave_chi_dis)
            ave_sum_chi_t = np.sum(ave_chi_dis_t)
            
            avmx = avmx / float(Nav * n_confi)
            avmy = avmy / float(Nav * n_confi)
            avmz = avmz / float(Nav * n_confi)
            avmag = avmag / float(Nav * n_confi)
            avE = avE / float(Nav * n_confi)
            effE = avE
            
            avmx_t = avmx_t / float(Nav * n_confi)
            avmy_t = avmy_t / float(Nav * n_confi)
            avmz_t = avmz_t / float(Nav * n_confi)
            avmag_t = avmag_t / float(Nav * n_confi)
            avE_t = avE_t / float(Nav * n_confi)
            effE_t = avE_t
            
            ave_chi_dis = ave_chi_dis / float(Nav * n_confi)
            ave_sum_chi = ave_sum_chi / float(Nav * n_confi)
            ave_chi_dis_t = ave_chi_dis_t / float(Nav * n_confi)
            ave_sum_chi_t = ave_sum_chi_t / float(Nav * n_confi)
            
            ave_helicity_j1_x = ave_helicity_j1_x / float(Nav * n_confi)
            ave_helicity_j1_y = ave_helicity_j1_y / float(Nav * n_confi)
            ave_helicity_j1_z = ave_helicity_j1_z / float(Nav * n_confi)
            ave_helicity_j2_x = ave_helicity_j2_x / float(Nav * n_confi)
            ave_helicity_j2_y = ave_helicity_j2_y / float(Nav * n_confi)
            ave_helicity_j2_z = ave_helicity_j2_z / float(Nav * n_confi)
            
            ave_helicity_j1_x_t = ave_helicity_j1_x_t / float(Nav * n_confi)
            ave_helicity_j1_y_t = ave_helicity_j1_y_t / float(Nav * n_confi)
            ave_helicity_j1_z_t = ave_helicity_j1_z_t / float(Nav * n_confi)
            ave_helicity_j2_x_t = ave_helicity_j2_x_t / float(Nav * n_confi)
            ave_helicity_j2_y_t = ave_helicity_j2_y_t / float(Nav * n_confi)
            ave_helicity_j2_z_t = ave_helicity_j2_z_t / float(Nav * n_confi)
            
            avsf_x = avsf_x / float(Nres * n_confi)
            avsf_y = avsf_y / float(Nres * n_confi)
            avsf_z = avsf_z / float(Nres * n_confi)
            avsf = avsf / float(Nres * n_confi)
            
            avsf_x_t = avsf_x_t / float(Nres * n_confi)
            avsf_y_t = avsf_y_t / float(Nres * n_confi)
            avsf_z_t = avsf_z_t / float(Nres * n_confi)
            avsf_t = avsf_t / float(Nres * n_confi)
            
            avsf_tildemag = avsf_tildemag / float(Nres * n_confi)
            
            ave_chiral = ave_chiral / float(Nres * n_confi)
            ave_tot_chiral = ave_tot_chiral / float(area * Nres * n_confi)
            
            # Write summary data to files
            write_summary_data(file_handles, ncount, Temp, avmag, avmx, avmy, avmz,
                             avmag_t, avmx_t, avmy_t, avmz_t, effE, effE_t,
                             ave_tot_chiral, ave_sum_chi, ave_sum_chi_t, area,
                             ave_helicity_j1_x, ave_helicity_j1_y, ave_helicity_j1_z,
                             ave_helicity_j2_x, ave_helicity_j2_y, ave_helicity_j2_z,
                             ave_helicity_j1_x_t, ave_helicity_j1_y_t, ave_helicity_j1_z_t,
                             ave_helicity_j2_x_t, ave_helicity_j2_y_t, ave_helicity_j2_z_t,
                             avsf, nx, ny)
            
            # Write all configuration files
            write_all_configuration_files(folder_name, ncount, sx, sy, sz, theta, phi,
                                        sx_t, sy_t, sz_t, ave_chiral, ave_chi_dis, ave_chi_dis_t,
                                        avsf, avsf_x, avsf_y, avsf_z,
                                        avsf_t, avsf_x_t, avsf_y_t, avsf_z_t,
                                        avsf_tildemag, nx, ny, Pi)
            
            t_end = time.time()
            if ncount % 10 == 0:  # Print every 10 steps
                print(f"  T={Temp:.4f}")
        
        # Close output files
        close_output_files(file_handles)
    
    print("\nSimulation completed!")


if __name__ == "__main__":
    main()