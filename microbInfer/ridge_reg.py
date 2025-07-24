import numpy as np

def compute_theta(Output, Input, D_lmd):
    """
    Compute the model parameters theta using the formula:
    theta = Output @ Input.T @ np.linalg.inv(Input @ Input.T + D_lmd)
    """
    return Output @ Input.T @ np.linalg.inv(Input @ Input.T + D_lmd)  # Calculate theta using the formula

