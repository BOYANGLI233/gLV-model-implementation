import numpy as np
from scipy.integrate import solve_ivp

def u_t(t, perturb):
    if perturb == 1:
        if t == 0:
            return 1
        else:
            return 0
    else:
        return 0

def f(t, x, mu, M, epsilon, perturb):
    u = u_t(t, perturb)
    dx_dt = x * (mu + M @ x + epsilon * u)

    return dx_dt

def gLV_simulation(t_span, t, x0, mu, M, epsilon, perturb):
    x = solve_ivp(f, t_span, x0, method='LSODA',args=(mu, M, epsilon, perturb), dense_output=True)
    simulated_data = x.sol(t)

    return simulated_data