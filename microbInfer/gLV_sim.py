import numpy as np
from scipy.integrate import solve_ivp

def u_t(t, perturb_start_end):
    if perturb_start_end is not None:
        num_p = perturb_start_end.shape[0]
        u = np.zeros(num_p)

        for p in range(num_p):
            start, end = perturb_start_end[p]
            if start <= t <= end:
                u[p] = 1

        return u
    else:
        return 0


def f(t, x, theta, perturb_start_end):
    num_x = len(x)
    mu = theta[:, num_x]
    M = theta[:, :num_x]
    dx_dt = x * (mu + M @ x)

    if perturb_start_end is not None:
        num_p = perturb_start_end.shape[0]

        if num_p == 1:
            epsilon = theta[:, num_x + 1]
            dx_dt += x * epsilon * u_t(t, perturb_start_end)
        else:
            epsilon = theta[:, num_x + 1:num_x + 1 + num_p]
            dx_dt += x * (epsilon @ u_t(t, perturb_start_end))

    return dx_dt

def gLV_simulation(t_span, num_t, x0, theta, x_intro, perturb_start_end=None):
    num_x = x0.shape[0]
    t_intro = np.unique(x_intro)
    num_sim = len(t_intro) 
    sim = [0] * num_sim
    sim_t = [0] * num_sim
 

    for s in range(num_sim):
        x0_sim = np.zeros(num_x)

        if s == num_sim - 1:
            sim_start_end = [t_intro[s], t_span[1]]  
        else:
            sim_start_end = [t_intro[s], t_intro[s + 1]]

        t = np.linspace(sim_start_end[0], sim_start_end[1], num_t)

        if s == 0:
            x0_sim = x0[:, s]
        else:
            active_x = np.where(x_intro == t_intro[s])[0]
            established_x = np.where(x_intro < t_intro[s])[0]
            x0_sim[active_x] = x0[:, s][active_x]
            x0_sim[established_x] = sim[s - 1][established_x, -1]

        x = solve_ivp(f, sim_start_end, x0_sim, method='LSODA', args=(theta, perturb_start_end), dense_output=True)

        if s == 0:
            sim[s] = x.sol(t)  # Store the simulation results for each species
            sim_t[s] = t
        else:
            sim[s] = x.sol(t)[:, 1:] # Skip the first time point to avoid duplication
            sim_t[s] = t[1:]

    x_sim = np.hstack(sim)
    t_sim = np.hstack(sim_t)

    return x_sim, t_sim