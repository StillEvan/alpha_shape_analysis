import numpy as np
import warnings

from scipy.spatial import Delaunay
#import alpha_shape_analysis
from alpha_shape_analysis.simplex_property_determination import *
from alpha_shape_analysis import alpha_hull
from alpha_shape_analysis import rejection_sampling
print(dir())
# Overall questions have the saving seperate or internal?

def cluster_alpha_shape_generation(spatial_coords):
    # spatial_coords should be an nx3 numpy array corresponding to X, Y, Z of a singular clusters ions

    d = len(spatial_coords[0])
    tri = Delaunay(spatial_coords, incremental=True)
    
    # Insert part where we save tri data to current hdf5 file if provided
    #
    #
    volume, radii, __, __ = get_simplex_properties(tri, d = d)

    # Might add a random initialization for the inputs and do it say 10 times.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        opt_alpha, d_glf_result, flag, message = alpha_heuristics.double_GLF_heuristic(radii, volume, tri, \
                                                 opt_method = 'ampgo', d = d)
    
    alpha_boolean = alpha_shape(tri, radii, opt_alpha)

    # Insert saving the volume, alpha, glf_fit parameters, mse attributes
    return tri, radii, volume, alpha_boolean, opt_alpha, message, flag

def cluster_ionic_abundance(test_coords, test_indices, tri, ion_types):
    # test coords should be a pandas dataframe
    comp_df = pd.DataFrame(0, index= 0, columns = Potential_Ions)

    for i in range(len(ion_types)):
        j = ion_Types[i]
        test_ion_indices = test_indices[test_coords['Ion Label'] == j]
        
        coords_test = test_coords.loc[test_coords['Ion Label'] == j,['X', 'Y', 'Z']].to_numpy()
            
        accept_coords, __, accept_boolean = rejection_sampling.rejection_sampling(tri, alpha_boolean, coords_test)

        comp_df.loc[i,j] = np.sum(accept_boolean)
    
    return comp_df