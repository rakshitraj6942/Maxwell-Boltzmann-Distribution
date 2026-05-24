from velocity_simulation import generate_velocities, compute_speeds
from matplotlib import pyplot as plt
import numpy as np

# Constants (K and kg)
T_gl = 300 ; m_gl = 4.65e-26
k_B = 1.380649e-23  # Boltzmann Constant (J/K)


def plots(T, m):
    '''
    Plots normalized histogram for speeds such that it approximates a probability density function for v (f(v)).
    '''
    # ---------- Velocities and Speed -------------
    velocity_comp = generate_velocities(N=100000, T=T, m=m)
    speeds = compute_speeds(velocities=velocity_comp)

    plt.figure(figsize=(10,5))

    # Histogram Plot
    plt.hist(speeds, bins='fd', density=True, alpha=0.7, label='Experimental Plot (np.random)') # Histogram plot
    plt.xlabel('Speed')
    plt.ylabel('Probability Density')

    # Theoretical Plot
    v = np.linspace(0, max(speeds), 500)
    f = 4*np.pi*(m/(2*np.pi*k_B*T))**(3/2)*(v**2)*np.exp(-m*v**2/(2*k_B*T))
    plt.plot(v, f, color='red', label='Maxewell-Boltzmann Distribution')
    
    plt.legend()
    plt.title(f'Comparison of Experimental and Theoretical Plots for N={len(speeds)}')
    plt.show()

plots(T=T_gl, m=m_gl)