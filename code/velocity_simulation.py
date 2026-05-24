import numpy as np
k_B = 1.380649e-23  # Boltzmann Constant (J/K)

# ------ Generation of Random Velocities (T: temperature, m: mass of the system) ---------
def generate_velocities(N, T: float , m: float):
    '''
    Returns N number of random velocity components (v_x, v_y, v_z) which follows Gaussian distributions consistent with thermal equilibrium.

    Consistency with thermal equilibrium means each velocity components follows a Gaussian distribution with mean 0 and variance equals to k_B*T/m.
    sigma = sqrt(k_B*T/m)

    P(v_x) = sqrt(m/2pi*k_B*T)*exp(-mv_x^2/2k_B*T)
    '''
    sigma = np.sqrt(k_B*T/m)
    
    v_x = np.random.normal(0, sigma, N)
    v_y = np.random.normal(0, sigma, N)
    v_z = np.random.normal(0, sigma, N)

    velocities = np.column_stack((v_x,v_y,v_z))

    return velocities

# ------ Computing speeds ---------
def compute_speeds(velocities):
    '''
    Returns the speed for each particle , v = sqrt(v_x^2 + v_y^2 + v_z^2)
    '''
    if isinstance(velocities, np.ndarray):
        return np.linalg.norm(velocities, axis=1)