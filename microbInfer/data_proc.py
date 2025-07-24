import numpy as np

def forward_diff_quotient(abundance, time):
    
    diff_quotient = np.empty((11,0)) # Initialize an empty array to store the difference quotients

    for i in range(abundance.shape[1] - 1):
        diff_quotient_i = (np.log(abundance[:, i + 1]) - np.log(abundance[:, i])) / (time[i + 1] - time[i])
        diff_quotient_i = diff_quotient_i.reshape(11, 1)  # Reshape to ensure it is a column vector

        for j in range(11):
            if np.isnan(diff_quotient_i[j, 0]):
                diff_quotient_i[j, 0] = 0
                
            if np.isinf(diff_quotient_i[j, 0]):
                diff_quotient_i[j, 0] = 0
                
        diff_quotient = np.hstack((diff_quotient, diff_quotient_i))
        
    return diff_quotient


def reorder_para_matrix(theta, current_order, desired_order):
    """
    Reorder the parameter matrix to match the order of species.
    
    Parameters:
    - theta: The parameter matrix (numpy array).
    - current_order: List of species names in the current order.
    - desired_order: List of species names in the desired order.

    Returns:
    - A numpy array with parameters reordered according to the species list.
    """

    new_order_indices = [current_order.index(species) for species in desired_order]
    new_M = theta[new_order_indices, :][:, new_order_indices]
    new_mu = theta[new_order_indices, -2]
    new_epsilon = theta[new_order_indices, -1]
    new_theta = np.hstack((new_M, new_mu.reshape(-1, 1), new_epsilon.reshape(-1, 1)))

    return new_theta