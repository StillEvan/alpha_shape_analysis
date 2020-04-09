import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# It might be beneficial to rewrite the plot_alpha and collect edges as well as the area, volume calc
# to build a network to make use of networkx functionality.
# It might also be simpler to make it be a homogenous alpha complex, k = 2, or k = 3.
# And use the smallest alpha that maintains purity, lets proceed with the idea of generating homogenous
# alpha complexes.

def sorted_tuple(a,b):
    return (a,b) if a < b else (b,a)

def plot_alpha_tri(tri, boolean_array, d = 2, reduce_dim = False, direction = 2):
    edges, __ = collect_alpha_edges(tri, boolean_array, d = d)
    
    x = np.array([])
    y = np.array([])
    z = np.array([])

    if d == 3:
        for (i,j) in edges:
            x = np.append(x, [tri.points[i, 0], tri.points[j, 0], np.nan])
            y = np.append(y, [tri.points[i, 1], tri.points[j, 1], np.nan])
            z = np.append(z, [tri.points[i, 2], tri.points[j, 2], np.nan])

        
        if reduce_dim:
            alpha_plot = plt.figure()
            if direction == 0:
                plt.plot(y, z, color='b', lw='1')
                plt.plot(tri.points[:, 1], tri.points[:, 2], color = 'k', linestyle = 'none', marker = '.')
            elif direction == 1:
                plt.plot(x, z, color='b', lw='1')
                plt.plot(tri.points[:, 0], tri.points[:, 2], color = 'k', linestyle = 'none', marker = '.')
            elif direction == 2:
                plt.plot(x, y, color='b', lw='1')
                plt.plot(tri.points[:, 0], tri.points[:, 1], color = 'k', linestyle = 'none', marker = '.')
            
            return alpha_plot
            
        else:
            alpha_plot = plt.figure()
            ax = alpha_plot.add_subplot(111, projection='3d')
            ax.plot3D(x, y, z, color='b', lw='1')
            ax.plot3D(tri.points[:, 0], tri.points[:, 1], tri.points[:, 2], color = 'k', linestyle = 'none', marker = '.')
        
        return alpha_plot, ax

    if d == 2:
        for (i,j) in edges:
            x = np.append(x, [tri.points[i, 0], tri.points[j, 0], np.nan])
            y = np.append(y, [tri.points[i, 1], tri.points[j, 1], np.nan])
        
        alpha_plot = plt.figure()
        plt.plot(x, y, color='b', lw='1')
        plt.plot(tri.points[:, 0], tri.points[:, 1], color = 'k', linestyle = 'none', marker = '.')

        return alpha_plot

def collect_alpha_edges(tri, boolean_array, d = 2):
    edges = set()
    vertices = set()
    # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    #for (i0, i1, i2, i3) in tri.simplices[boolean_array]:

    if d == 2:
        for (i0, i1, i2) in tri.simplices[boolean_array]:
            vertices.add(i0)
            vertices.add(i1)
            vertices.add(i2)

            edges.add(sorted_tuple(i0,i1))
            edges.add(sorted_tuple(i0,i2))
            edges.add(sorted_tuple(i1,i2))
        
    if d == 3:
        for (i0, i1, i2, i3) in tri.simplices[boolean_array]:
            vertices.add(i0)
            vertices.add(i1)
            vertices.add(i2)
            vertices.add(i3)

            edges.add(sorted_tuple(i0,i1))
            edges.add(sorted_tuple(i0,i2))
            edges.add(sorted_tuple(i0,i3))
            edges.add(sorted_tuple(i1,i2))
            edges.add(sorted_tuple(i1,i3))
            edges.add(sorted_tuple(i2,i3))

    return edges, vertices

def alpha_shape(tri, simplex_radius, alpha):
    alpha_boolean = np.array(simplex_radius) < alpha
    return alpha_boolean
    