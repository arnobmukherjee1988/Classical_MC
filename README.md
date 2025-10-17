# Monte Carlo Simulation for Rashba Spin-Orbit Coupled Systems

Monte Carlo simulation for studying twisted magnetic skyrmions.

### Background  
The starting point is an anisotropic Kondo–lattice Hamiltonian that combines ordinary exchange coupling ($J_1$) and an additional anisotropic Kondo interaction ($J_2$):

$$\mathcal{H} =- \sum_{i,\gamma,\sigma} \left( t_\gamma\, c_{i\sigma}^\dagger c_{i+\gamma,\sigma} + \text{H.c.} \right)+ J_1 \sum_i \mathbf{S}_i \cdot \boldsymbol{\tau}_i+ J_2 \sum_i (\mathbf{S}_i \times \boldsymbol{\tau}_i) \cdot \hat{z}.$$

The $J_2$ term introduces an in-plane twist between localized and itinerant spins, producing an **effective local spin**


$\mathbf{S}_i^{\text{eff}} = \mathbf{S}_i + \frac{J_2}{J_1}\, \hat{z} \times \mathbf{S}_i$

which rotates the spin quantization axis by an angle proportional to the ratio  
$\alpha = J_2 / J_1$.

This transformation gives rise to a finite helicity  
$\gamma = -\arctan(\alpha)$,  
so the skyrmion’s internal rotation directly reflects the balance between conventional and anisotropic Kondo couplings.  

## Key Parameters

Edit in `input_parameters.py`:

- `nx`, `ny` - Lattice size (default: 40x40)
- `Neq`, `Nav` - MC steps for equilibration and averaging
- `alpha` - Rashba coupling parameter
- `min_temp`, `max_temp` - Temperature range
- `B_zmn_min`, `B_zmn_max` - Applied magnetic field range

## Output

Creates `Bzmn_X.XXXXX/` folders with:
- `mag_vs_T.txt`, `energy_vs_T.txt`, etc. - Observables vs temperature
- `XXX.dat` files - Spin configurations and structure factors

## Results

<!-- ![Skyrmion Configuration 1](result_plot.png) ![Skyrmion Configuration 2](figure2.png) -->

![Skyrmion Configuration 1](result_plot.png)

Skyrmion lattice at low temperature; (a) $J2/J1 = 0$, and (b) $J2/J1 = 0.9$. Arrows show in-plane spins, colors show out-of-plane component.

## Speed

Code uses Numba JIT compilation for 20-100x speedup. First run compiles functions (~30 sec), then runs fast.

## Files

- `input_parameters.py` - All parameters
- `main_simulation.py` - Main program
- `monte_carlo.py` - MC algorithm
- `energy_functions.py` - Hamiltonian Energy calculations
- `observables.py` - Observable calculations
- `file_io_functions.py` - File operations
- `plot_lowest_temperature.py` - Plotting Spin configuration and structure factor
