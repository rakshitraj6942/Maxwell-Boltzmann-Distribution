import numpy as np
import scipy.integrate as scint
from velocity_simulation import generate_velocities, compute_speeds

# Constants
k_B = 1.380649e-23  # Boltzmann Constant (J/K)
T_gl = 300 # Temperature (K)
m_gl = 4.65e-26 # Mass of the System (kg)

# Mean Speed

def mean_speed(speeds):
    '''
    Returns the Mean speed of the input speeds array.

    Mean Speed = <v> = 0∫∞ vf(v)dv [Definition]

    Also, <v> = ∑v_i / n (where n is the length of speeds array)
    '''
    return (np.sum(speeds))/len(speeds)

def integration_mean_speeds(speeds):
    '''
    Computing mean speed using the integration definition of it.

    <v> = 0∫∞ vf(v)dv, where f(v) is Maxewell-Boltzmann Speed Distribution.

    As we only have the values of speed, we will first approximate our probability density function f(v) (by histogram) and then compute the integration.
    '''

    hist, bin_edges = np.histogram(
        speeds,
        bins='fd',
        density=True,
    )
    
    # Trapezoid Rule (approximation of integration)
    # 0∫∞ vf(v)dv = 0∫∞ g(v)dv = ∑ (g_i + g_(i+1))/2 * Δv (Sum of Trapezoids)

    v_centers = (bin_edges[:-1] + bin_edges[1:])/2

    return np.trapezoid(v_centers * hist , v_centers)

def most_prob_speed(speeds):
    '''
    Computes the most probale speed of the input speeds array.

    Uses the technique that the most probable speed is the speed that occurs the most number of times. So we use histogram to divide speeds into bins, and the bins having the greatest height will contain the v_mp, so approximaltely v_mp is equal to the speed at middle of that bin.
    '''
    hist, bin_edges = np.histogram(
        speeds,
        bins='fd', 
        density=True
    )

    peak_bin = np.argmax(hist)

    return (bin_edges[peak_bin] + bin_edges[peak_bin+1])/2

def rms_speed(speeds):
    '''
    Computes the rms speed of the input speeds array.

    v_rms = sqrt[ (∑v_i ^2) / n ]
    '''
    return np.sqrt((np.sum(speeds**2))/len(speeds))