import numpy as np
from scipy.integrate import solve_ivp

def u_t(t, perturb, perturb_start_end):
    if perturb == 1:
        if perturb_start_end[0] <= t and t <= perturb_start_end[1]:
            return 1
        else:
            return 0
    else:
        return 0

def f(t, x, mu, M, epsilon, perturb, perturb_start_end):
    u = u_t(t, perturb, perturb_start_end)
    dx_dt = x * (mu + M @ x + epsilon * u)

    return dx_dt

def gLV_simulation(t_span, t, x0, mu, M, epsilon, perturb, perturb_start_end=(0, 0)):
    x = solve_ivp(f, t_span, x0, method='LSODA', args=(mu, M, epsilon, perturb, perturb_start_end), dense_output=True)
    simulated_data = x.sol(t)

    return simulated_data