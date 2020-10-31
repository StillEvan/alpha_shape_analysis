import numpy as np
import warnings
import pandas as pd
from scipy.spatial import Delaunay

from alpha_shape_analysis.simplex_property_determination import *
from alpha_shape_analysis.alpha_hull import *
from alpha_shape_analysis.alpha_heuristics import *
from alpha_shape_analysis.rejection_sampling import *

def cluster_alpha_shape_generation(spatial_coords, n_randomizations = 0):
    # spatial_coords should be an nx3 numpy array corresponding to X, Y, Z of a singular clusters ions

    d = len(spatial_coords[0])
    tri = Delaunay(spatial_coords, incremental=True)
    
    # Insert part where we save tri data to current hdf5 file if provided
    #
    #
    volume, radii, __, __ = get_simplex_properties(tri, d = d)

    # Might add a random initialization for the inputs and do it say 10 times.
    if n_randomizations != 0:
        counter = 0
        aic = 0
        while counter < n_randomizations:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                temp_alpha, temp_d_glf_result, temp_n_inflection, temp_flag, temp_message = double_GLF_heuristic(radii, volume, tri, \
                                                opt_method = 'ampgo', d = d, a_mid = np.random.sample())

            if aic > temp_d_glf_result.aic:
                print(str(temp_d_glf_result.aic) + ' is lower than prior aic')
                opt_alpha = temp_alpha
                d_glf_result = temp_d_glf_result
                n_inflection = temp_n_inflection
                flag = temp_flag
                message = temp_message
                aic = temp_d_glf_result.aic
            counter += 1
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt_alpha, d_glf_result, n_inflection, flag, message = double_GLF_heuristic(radii, volume, tri, \
                                                opt_method = 'ampgo', d = d)
    alpha_boolean = alpha_shape(tri, radii, opt_alpha)

    # Insert saving the volume, alpha, glf_fit parameters, mse attributes
    return tri, radii, volume, alpha_boolean, opt_alpha, d_glf_result, message, flag

def cluster_ionic_abundance(test_coords, test_indices, tri, alpha_boolean, ion_types):
    # test coords should be a pandas dataframe
    comp_df = pd.DataFrame(columns = ion_types)
    
    for i in range(len(ion_types)):
        j = ion_types[i]
        print('----Processing ' + j)
        test_ion_indices = test_indices[test_coords['Ion Label'] == j]
        
        coords_test = test_coords.loc[test_coords['Ion Label'] == j,['X', 'Y', 'Z']].to_numpy()
        print(len(coords_test))
        accept_coords, __, accept_boolean = rejection_sampling(tri, alpha_boolean, coords_test)
        print(np.sum(accept_boolean))
        comp_df.loc[0,j] = np.sum(accept_boolean)

        if i == 0:
            accepted_indices = test_ion_indices[accept_boolean]
            print(accepted_indices)
        else:
            accepted_indices = np.append(accepted_indices, test_ion_indices[accept_boolean])

    return comp_df, accepted_indices