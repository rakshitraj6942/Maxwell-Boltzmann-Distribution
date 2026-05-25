import numpy as np
from matplotlib import pyplot as plt
from velocity_simulation import generate_velocities, compute_speeds

k_B = 1.380649e-23  # Boltzmann Constant (J/K)
m_gl = 4.65e-26

def diff_temp(temps, m: float):
    '''
    Plots Maxewell Distribution and Theoretical Plots for different temperatures.

    Note : temps must be an array of different temperatures (try to keep the size of T small).
    '''
    
    plt.figure(figsize=(10,5))
    t = min(temps)
    plt.hist(compute_speeds(velocities=generate_velocities(N=100000, T=t, m=m)), bins='fd', density=True, alpha=0.6, label=f'Histogram for T={t}K')
    
    for T in temps:
        speeds = compute_speeds(velocities=generate_velocities(N=100000, T=T, m=m))
        v = np.linspace(0, max(speeds), 500)
        f = 4*np.pi*(m/(2*np.pi*k_B*T))**(3/2)*(v**2)*np.exp(-m*v**2/(2*k_B*T))
        plt.plot(v, f, label=f'T = {T} K')

    plt.legend()
    plt.xlabel('Speeds (m/s)')
    plt.ylabel('Probability Density')
    plt.title('Temperature Dependence of Velocity Distribution')
    plt.show()

temp_arr = [200, 300, 400, 500]
diff_temp(temp_arr, m_gl)