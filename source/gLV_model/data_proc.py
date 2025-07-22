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