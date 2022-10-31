import numpy as np
import matplotlib.pyplot as plt

def basis_w(w, order=1, gamma=1):
    return 1/(np.math.factorial(order)*(w * 1j / gamma + 1)**(order + 1))

def basis(t, order=1, gamma=1):
    return gamma**(order+1) / np.math.factorial(order) * t**order * np.exp(-gamma*t) if t > 0 else 0

def rates_ss(W,b=-1,Tmax=1000):  # set inputs here

    '''
    compute steady-state mean field rates through Euler step
    :param W: weight matrix
    :return: steady-state rates with transfer functions and cellular/synaptic parameters defined in params.py and phi.py
    '''

    N = W.shape[0]

    dt = 0.1
    
    a = 1.
    a2 = a ** 2

    r = np.zeros(N)
    s_dummy = np.zeros(N)
    s = np.zeros(N)
    b = -1

    r_vec = np.zeros((N, Tmax))
    for i in range(Tmax):
        s_dummy += dt * (-2 * a * s_dummy - a2 * s) + r * a2 * dt
        s += dt * s_dummy

        g = W.dot(s) + b
        r = np.exp(g)
        r_vec[:, i] = r

    return r

def linear_response_function(w, W, phi_ss):
    N = W.shape[0]
    return np.linalg.inv(np.eye(N) - phi_ss.reshape(-1,1) * W * basis_w(w, order=1, gamma=1)) # phi_ss is the first derivative


def cross_covariance(w, W, phi_ss):
    Delta = linear_response_function(w, W, phi_ss)
    return (Delta@np.diag(phi_ss) @np.conjugate(Delta).T) # phi_ss is phi_0
#     return phi_ss * Delta @ np.conjugate(Delta).T # should be equivelent to this

def plot_cov_time(weight, Nw=200):
    """ Calculate Tree-level cross-covariance in frequency domain
     Args:
         weight [2d array]: weight matrix of the network
         Nw [int]: number of frequency points sampled
     Return:
         cross-covariance, time range
    """
    phi_steady_state = rates_ss(weight,1000)
    dt = 0.1
    t0 = -10
    w_range = np.fft.fftfreq(Nw, dt)
#     phase_factor =  [np.exp(-1j*2*np.pi*w*t0) for w in w_range]
    phase_factor_w = lambda w: np.exp(1j*2*np.pi*w*t0)
    c = np.array([cross_covariance(w, weight, phi_steady_state) * phase_factor_w(w) for w in w_range])
    num_ele = weight.shape[0]**2
    c = np.transpose(np.array(c),[0, 2, 1]).reshape(Nw, num_ele).T # swap last two axes from transpose function
    return c, np.arange(t0, -t0, dt)
