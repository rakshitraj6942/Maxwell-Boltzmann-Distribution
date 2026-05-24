from velocity_simulation import generate_velocities, compute_speeds
from matplotlib import pyplot as plt
import numpy as np

# Temperature and Mass (K and kg)
T = 300 ; m = 4.65e-26

# ---------- Velocities and Speed -------------
velocity_comp = generate_velocities(N=10000, T=T, m=m)
speeds = compute_speeds(velocities=velocity_comp)

# _____ Histogram Plot _______

def histogram_pdf(speeds: np.ndarray):
    '''
    Plots normalized histogram for speeds such that it approximates a probability density function for v (f(v)).
    '''
    plt.figure(figsize=(10,5))
    plt.hist(speeds, bins='fd', density=True)
    plt.xlabel('Speed')
    plt.ylabel('Probability Density')
    plt.show()

histogram_pdf(speeds)