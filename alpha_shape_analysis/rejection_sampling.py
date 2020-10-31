import numpy as np
def rejection_sampling(tri, alpha_boolean, test_data):
    # Might be beneficial to have a seperate section for rejection sampling types
    acceptance_boolean = np.zeros(len(test_data), dtype=bool)

    found_simplices = tri.find_simplex(test_data)
    pass_one_accept = np.argwhere(found_simplices != -1)
    pass_two_accept = pass_one_accept[np.argwhere(alpha_boolean[found_simplices[found_simplices != -1]] == True)]
    acceptance_boolean[pass_two_accept] = 1
      
    return test_data[acceptance_boolean], test_data[np.invert(acceptance_boolean)], acceptance_boolean