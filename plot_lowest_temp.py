import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import os
from matplotlib import rc

plt.style.use("classic")
rc('axes', edgecolor='k')

Pi = np.pi

# Arrow parameters for spin configuration
separation = 4
arrow_length = 1.5
arrow_width = 0.60
arrow_head_width = 3.0 * arrow_width
arrow_head_length = 1.5 * arrow_head_width


def plot_spin_configuration(filenum, output_folder, spin_type):
    
    # Plot spin configuration from filenum.dat
    
    filepath = str(filenum) + '.dat'
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found, skipping...")
        return
    
    # Read data
    x, y = np.genfromtxt(filepath, usecols=(0, 1), unpack=True)
    sx, sy, sz = np.genfromtxt(filepath, usecols=(3, 4, 5), unpack=True)
    
    # Determine grid size
    num_of_x_data = 1
    for i in range(len(y) - 1):
        if np.abs(y[i+1] - y[i]) <= 1e-8:
            num_of_x_data = num_of_x_data + 1
        else:
            break
    
    num_of_y_data = np.int64(np.size(y) / num_of_x_data)
    nx = num_of_x_data
    ny = num_of_y_data
    
    # Reshape to 2D arrays
    sx = np.reshape(sx, (nx, ny), order='F')
    sy = np.reshape(sy, (nx, ny), order='F')
    sz = np.reshape(sz, (nx, ny), order='F')
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    mag_sx_sy = np.sqrt(sx**2 + sy**2)
    
    ax = plt.axes([0.01, 0.03, 0.85, 0.85])
    fig.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(-separation + separation * 0, separation * nx)
    ax.set_ylim(-separation + separation * 0, separation * ny)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Color normalization for sz component
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.cm.gnuplot
    
    # Plot arrows
    for ix in range(nx):
        for iy in range(ny):
            max_mag = np.max(mag_sx_sy)
            if max_mag > 0:
                arrow_w = 0.60 * mag_sx_sy[ix, iy] / max_mag
            else:
                arrow_w = 0.01
            
            arrow_hw = 3.0 * arrow_w
            arrow_hl = 1.5 * arrow_hw
            
            ax.arrow(separation * ix - arrow_length * sx[ix, iy], 
                    separation * iy - arrow_length * sy[ix, iy],
                    arrow_length * sx[ix, iy], 
                    arrow_length * sy[ix, iy],
                    width=arrow_w, 
                    ec=cmap(norm(sz[ix, iy])), 
                    fc=cmap(norm(sz[ix, iy])),
                    head_length=arrow_hl, 
                    head_width=arrow_hw)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = plt.axes([0.87, 0.03, 0.03, 0.85])
    cax.tick_params(labelsize=30)
    fig.colorbar(sm, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cax.set_yticklabels([r'$-1.0$', r'$-0.5$', r'$0.0$', r'$0.5$', r'$1.0$'])
    
    # Save figure
    output_path = os.path.join(output_folder, f'{spin_type}_spin_{filenum}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_structure_factor(filenum, output_folder, spin_type):
    
    # Plot structure factor from filenum.dat
    
    filepath = str(filenum) + '.dat'
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found, skipping...")
        return
    
    # Read data
    x, y, z = np.genfromtxt(filepath, usecols=(0, 1, 2), unpack=True)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)
    
    # Determine grid size
    num_of_x_data = 1
    for i in range(len(y) - 1):
        if np.abs(y[i+1] - y[i]) <= 1e-8:
            num_of_x_data = num_of_x_data + 1
        else:
            break
    
    num_of_y_data = np.int64(np.size(y) / num_of_x_data)
    
    # Reshape and transpose for proper orientation
    Z = np.transpose(np.reshape(z, (num_of_x_data, num_of_y_data), order="C"))
    
    # Flip for correct orientation
    Z1 = np.zeros((num_of_x_data, num_of_y_data))
    for iy in range(num_of_y_data):
        for ix in range(num_of_x_data):
            jy = ix
            jx = num_of_y_data - iy - 1
            Z1[jx, jy] = Z[ix, iy]
    
    # Create custom colormap
    colors = [u'white', u'blue', u'red', u'orange']
    cmap_name = 'my_list'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=1000)
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes([0.01, 0.03, 0.85, 0.85])
    fig.gca().set_aspect('equal', adjustable='box')
    
    # Set spine widths
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    ax.grid(linewidth=1.5)
    
    # Plot structure factor
    im = ax.imshow(Z1, cmap=cmap, aspect="equal", interpolation='bilinear',
                  extent=[xmin, xmax, ymin, ymax], vmin=zmin, vmax=zmax)
    
    # Set axis ticks and labels
    ax.set_xticks([-Pi, -Pi/2.0, 0, Pi/2.0, Pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", 
                       r"$\frac{\pi}{2}$", r"$\pi$"], fontsize=35)
    ax.set_yticks([-Pi, -Pi/2.0, 0, Pi/2.0, Pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", 
                       r"$\frac{\pi}{2}$", r"$\pi$"], fontsize=35)
    ax.set_xlabel("$k_x$", fontstyle='oblique', weight='bold', fontsize=35)
    ax.set_ylabel("$k_y$", fontstyle='oblique', weight='bold', 
                 rotation=0, fontsize=35)
    
    # Add colorbar
    norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = plt.axes([0.88, 0.03, 0.03, 0.852])
    cb = plt.colorbar(sm, cax=cax)
    cax.tick_params(labelsize=30)
    
    # Save figure
    output_path = os.path.join(output_folder, f'{spin_type}_sf_{filenum}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def process_all_folders():
    
    # Process all Bzmn folders and create plots for lowest temperature.
    
    # Get all Bzmn folders
    folders = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('Bzmn_'):
            folders.append(item)
    
    folders.sort()
    
    if len(folders) == 0:
        print("No Bzmn folders found in current directory!")
        return
    
    print(f"Found {len(folders)} Bzmn folders")
    print("="*60)
    
    # Process each folder
    for folder in folders:
        
        # Create output directories
        real_spin_folder = os.path.join(folder, 'real_spin')
        tilde_spin_folder = os.path.join(folder, 'tilde_spin')
        
        os.makedirs(real_spin_folder, exist_ok=True)
        os.makedirs(tilde_spin_folder, exist_ok=True)
        
        # Change to folder directory
        os.chdir(folder)
        
        # Plot real spin configuration (150.dat)
        plot_spin_configuration(150, 'real_spin', 'real')
        
        # Plot real spin structure factor (250.dat)
        plot_structure_factor(250, 'real_spin', 'real')
        
        # Plot tilde spin configuration (1050.dat)
        plot_spin_configuration(1050, 'tilde_spin', 'tilde')
        
        # Plot tilde spin structure factor (2050.dat)
        plot_structure_factor(2050, 'tilde_spin', 'tilde')
        
        # Return to parent directory
        os.chdir('..')
        
        print(f"{folder}")
    


if __name__ == "__main__":
    print("="*60)
    print("Lowest Temperature Spin Configuration and Structure Factor Plot")
    print("="*60)    
    process_all_folders()
