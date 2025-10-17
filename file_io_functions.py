import numpy as np
import os


def setup_output_directories(folder_name):
    # Create output directory structure
    os.makedirs(folder_name, exist_ok=True)


def open_output_files(folder_name):
    # Open all output files for writing, return dictionary of file handles
    file_dict = {
        'mag': open(f"{folder_name}/mag_vs_T.txt", 'w'),
        'mag_t': open(f"{folder_name}/mag_vs_T_tilda.txt", 'w'),
        'energy': open(f"{folder_name}/energy_vs_T.txt", 'w'),
        'energy_t': open(f"{folder_name}/energy_vs_T_tilda.txt", 'w'),
        'chirality': open(f"{folder_name}/chirality_vs_T.txt", 'w'),
        'skyr_den': open(f"{folder_name}/skyr_den_vs_T.txt", 'w'),
        'skyr_den_t': open(f"{folder_name}/skyr_den_vs_T_tilda.txt", 'w'),
        'helicity': open(f"{folder_name}/helicity_vs_T.txt", 'w'),
        'helicity_t': open(f"{folder_name}/helicity_vs_T_tilda.txt", 'w'),
        'stagg_mag': open(f"{folder_name}/stagg_mag_peaks.txt", 'w')
    }
    return file_dict


def close_output_files(file_dict):
    # Close all open file handles
    for f_handle in file_dict.values():
        f_handle.close()


def write_summary_data(file_dict, ncount, Temp, avmag, avmx, avmy, avmz,
                       avmag_t, avmx_t, avmy_t, avmz_t, effE, effE_t,
                       ave_tot_chiral, ave_sum_chi, ave_sum_chi_t, area_val,
                       ave_helicity_j1_x, ave_helicity_j1_y, ave_helicity_j1_z,
                       ave_helicity_j2_x, ave_helicity_j2_y, ave_helicity_j2_z,
                       ave_helicity_j1_x_t, ave_helicity_j1_y_t, ave_helicity_j1_z_t,
                       ave_helicity_j2_x_t, ave_helicity_j2_y_t, ave_helicity_j2_z_t,
                       avsf, nx_val, ny_val):
    # Write summary data (single line per temperature) to files
    
    # Magnetization
    file_dict['mag'].write(f"{ncount} {Temp} {avmag} {avmx} {avmy} {avmz}\n")
    file_dict['mag_t'].write(f"{ncount} {Temp} {avmag_t} {avmx_t} {avmy_t} {avmz_t}\n")
    
    # Energy
    file_dict['energy'].write(f"{ncount} {Temp} {effE}\n")
    file_dict['energy_t'].write(f"{ncount} {Temp} {effE_t}\n")
    
    # Chirality
    file_dict['chirality'].write(f"{ncount} {Temp} {ave_tot_chiral} {np.abs(ave_tot_chiral)}\n")
    
    # Skyrmion density
    file_dict['skyr_den'].write(f"{ncount} {Temp} {ave_sum_chi} {ave_sum_chi/area_val}\n")
    file_dict['skyr_den_t'].write(f"{ncount} {Temp} {ave_sum_chi_t} {ave_sum_chi_t/area_val}\n")
    
    # Helicity
    file_dict['helicity'].write(f"{ncount} {Temp} {ave_helicity_j1_x} {ave_helicity_j1_y} "
                                f"{ave_helicity_j1_z} {ave_helicity_j2_x} {ave_helicity_j2_y} "
                                f"{ave_helicity_j2_z}\n")
    file_dict['helicity_t'].write(f"{ncount} {Temp} {ave_helicity_j1_x_t} {ave_helicity_j1_y_t} "
                                  f"{ave_helicity_j1_z_t} {ave_helicity_j2_x_t} {ave_helicity_j2_y_t} "
                                  f"{ave_helicity_j2_z_t}\n")
    
    # Staggered magnetization peaks
    file_dict['stagg_mag'].write(f"{ncount} {Temp} {avsf[nx_val//2, ny_val//2]} {avsf[nx_val, ny_val]} "
                                 f"{avsf[3*nx_val//4, 3*ny_val//4]} {avsf[nx_val, ny_val//2]} "
                                 f"{avsf[nx_val//4, 3*ny_val//4]} {avsf[3*nx_val//4, ny_val]} "
                                 f"{avsf[3*nx_val//4, ny_val//2]}\n")


def write_spin_configuration(folder_name, ncount, sx, sy, sz, theta, phi, nx_val, ny_val):
    # Write real spin configuration to file (100+ncount.dat)
    filenum = ncount + 100
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for iy in range(1, ny_val + 1):
            for ix in range(1, nx_val + 1):
                i = (iy - 1) * nx_val + ix - 1
                f.write(f"{ix} {iy} {i} {sx[i]} {sy[i]} {sz[i]} {theta[i]} {phi[i]}\n")


def write_tilde_spin_configuration(folder_name, ncount, sx_t, sy_t, sz_t, nx_val, ny_val):
    # Write tilde spin configuration to file (1000+ncount.dat)
    filenum = ncount + 1000
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for iy in range(1, ny_val + 1):
            for ix in range(1, nx_val + 1):
                i = (iy - 1) * nx_val + ix - 1
                mag_t = np.sqrt(sx_t[i]**2 + sy_t[i]**2 + sz_t[i]**2)
                f.write(f"{ix} {iy} {i} {sx_t[i]} {sy_t[i]} {sz_t[i]} {mag_t}\n")
            f.write("\n")


def write_structure_factor(folder_name, ncount, avsf, avsf_x, avsf_y, avsf_z, 
                           nx_val, ny_val, Pi):
    # Write real spin structure factor to file (200+ncount.dat)
    filenum = ncount + 200
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for j in range(0, ny_val + 1):
            for i in range(0, nx_val + 1):
                kx = -Pi + (float(i) * 2.0 * Pi) / float(nx_val)
                ky = -Pi + (float(j) * 2.0 * Pi) / float(ny_val)
                f.write(f"{kx} {ky} {avsf[i, j]} {avsf_x[i, j]} {avsf_y[i, j]} {avsf_z[i, j]}\n")
            f.write("\n")


def write_tilde_structure_factor(folder_name, ncount, avsf_t, avsf_x_t, avsf_y_t, avsf_z_t,
                                 nx_val, ny_val, Pi):
    # Write tilde spin structure factor to file (2000+ncount.dat)
    filenum = ncount + 2000
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for j in range(0, ny_val + 1):
            for i in range(0, nx_val + 1):
                kx = -Pi + (float(i) * 2.0 * Pi) / float(nx_val)
                ky = -Pi + (float(j) * 2.0 * Pi) / float(ny_val)
                f.write(f"{kx} {ky} {avsf_t[i, j]} {avsf_x_t[i, j]} {avsf_y_t[i, j]} {avsf_z_t[i, j]}\n")
            f.write("\n")


def write_chirality(folder_name, ncount, ave_chiral, nx_val, ny_val):
    # Write chirality distribution to file (300+ncount.dat)
    filenum = ncount + 300
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for iy in range(1, ny_val + 1):
            for ix in range(1, nx_val + 1):
                f.write(f"{ix} {iy} {ave_chiral[ix-1, iy-1]}\n")
            f.write("\n")


def write_skyrmion_density(folder_name, ncount, ave_chi_dis, nx_val, ny_val):
    # Write real spin skyrmion density to file (400+ncount.dat)
    filenum = ncount + 400
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for iy in range(1, ny_val + 1):
            for ix in range(1, nx_val + 1):
                f.write(f"{ix} {iy} {ave_chi_dis[ix-1, iy-1]}\n")
            f.write("\n")


def write_tilde_skyrmion_density(folder_name, ncount, ave_chi_dis_t, nx_val, ny_val):
    # Write tilde spin skyrmion density to file (4000+ncount.dat)
    filenum = ncount + 4000
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for iy in range(1, ny_val + 1):
            for ix in range(1, nx_val + 1):
                f.write(f"{ix} {iy} {ave_chi_dis_t[ix-1, iy-1]}\n")
            f.write("\n")


def write_tilde_magnitude_structure_factor(folder_name, ncount, avsf_tildemag, 
                                           nx_val, ny_val, Pi):
    # Write tilde spin magnitude structure factor to file (5000+ncount.dat)
    filenum = ncount + 5000
    filepath = f"{folder_name}/{filenum}.dat"
    
    with open(filepath, 'w') as f:
        for j in range(0, ny_val + 1):
            for i in range(0, nx_val + 1):
                kx = -Pi + (float(i) * 2.0 * Pi) / float(nx_val)
                ky = -Pi + (float(j) * 2.0 * Pi) / float(ny_val)
                f.write(f"{kx} {ky} {avsf_tildemag[i, j]}\n")
            f.write("\n")


def write_all_configuration_files(folder_name, ncount, sx, sy, sz, theta, phi,
                                  sx_t, sy_t, sz_t, ave_chiral, ave_chi_dis, ave_chi_dis_t,
                                  avsf, avsf_x, avsf_y, avsf_z,
                                  avsf_t, avsf_x_t, avsf_y_t, avsf_z_t,
                                  avsf_tildemag, nx_val, ny_val, Pi):
    # Write all configuration files for a given temperature point
    write_spin_configuration(folder_name, ncount, sx, sy, sz, theta, phi, nx_val, ny_val)
    write_tilde_spin_configuration(folder_name, ncount, sx_t, sy_t, sz_t, nx_val, ny_val)
    write_structure_factor(folder_name, ncount, avsf, avsf_x, avsf_y, avsf_z, nx_val, ny_val, Pi)
    write_tilde_structure_factor(folder_name, ncount, avsf_t, avsf_x_t, avsf_y_t, avsf_z_t, nx_val, ny_val, Pi)
    write_chirality(folder_name, ncount, ave_chiral, nx_val, ny_val)
    write_skyrmion_density(folder_name, ncount, ave_chi_dis, nx_val, ny_val)
    write_tilde_skyrmion_density(folder_name, ncount, ave_chi_dis_t, nx_val, ny_val)
    write_tilde_magnitude_structure_factor(folder_name, ncount, avsf_tildemag, nx_val, ny_val, Pi)