import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
from scipy.signal import find_peaks
from lmfit import Model
import warnings
#from multiprocessing import Pool

from alpha_shape_analysis import alpha_hull

# Support functions for graph heuristics
def network_from_collection(edges, vertices):
    
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    return G

def generate_alpha_network(tri, boolean_array, d = 2):

    G = nx.Graph()
    if d == 2:
        for (i0, i1, i2) in tri.simplices[boolean_array]:
            G.add_node(i0)
            G.add_node(i1)
            G.add_node(i2)

            G.add_edge(i0,i1)
            G.add_edge(i0,i2)
            G.add_edge(i1,i2)

    if d == 3:
        for (i0, i1, i2, i3) in tri.simplices[boolean_array]:
            G.add_node(i0)
            G.add_node(i1)
            G.add_node(i2)
            G.add_node(i3)

            G.add_edge(i0,i1)
            G.add_edge(i0,i2)
            G.add_edge(i0,i3)
            G.add_edge(i1,i2)
            G.add_edge(i1,i3)
            G.add_edge(i2,i3)
    
    return G

# Graph heuristic
def graph_heuristic(tri, radii, alpha_list, d = 2, heuristic = 'H2'):
    """
    H1 denotes a homogenous simplex containing all vertices
    H2 denotes a homogenous simplex that is fully connected and contains all vertices
    """
    if heuristic == 'H1':
        n_contained_vertices = []
        for i in range(len(alpha_list)):
            j = alpha_list[i]
            alpha_boolean = alpha_hull.alpha_shape(tri, radii, j)
            __, vertices = alpha_hull.collect_alpha_edges(tri, alpha_boolean, d = d)
            n_contained_vertices.append(len(vertices))
        
        optimal_alpha = alpha_list[np.argmax(n_contained_vertices)]

    elif heuristic == 'H2':
        # Component counting realllllllly slows shit down
        n_contained_vertices = []
        n_components = []
        n_simplices = []
        for i in range(len(alpha_list)):
            j = alpha_list[i]
            alpha_boolean = alpha_shape(tri, radii, j)

            edges, vertices = collect_alpha_edges(tri, alpha_boolean, d = d)
            G_alpha = network_from_collection(edges, vertices)
            n_components.append(nx.number_connected_components(G_alpha))
            n_contained_vertices.append(len(vertices))
            n_simplices.append(sum(alpha_boolean))
        vertex_pos = np.argmax(n_contained_vertices)
        components_pos = np.argmin(n_components[vertex_pos:]) + vertex_pos
        optimal_alpha = alpha_list[components_pos]
        
        return optimal_alpha, components_pos, n_components, n_contained_vertices, n_simplices
    else:
        print('Invalid heuristic')
    
    return optimal_alpha

# Support functions for regression heuristics
def glf(x, a1, a2, c1, b1, q1, v1):
    return a1 + np.divide(a2 - a1, np.power(c1 + q1*np.exp(-b1*x), 1/v1))

def double_glf(x, a1, a2, a3, c1, c2, b1, b2, q1, q2, v1, v2):
    # Note a1, a2, a3 are not necessarily the asymptotes
    # By construction a1 is the lowest asymptote, a3 is the highest asymptote, 
    # and a3 - a2 establishes the middle asymptote
    #print(a1, a2, a3, c1, c2, b1, b2, q1, q2, v1, v2)
    return a1 + np.divide(a2 - a1, np.power(c1 + q1*np.exp(-b1*x), 1/v1)) + np.divide(a3 - a2, np.power(c2 + q2*np.exp(-b2*x), 1/v2))

def double_glf_dx(x, a1, a2, a3, c1, c2, b1, b2, q1, q2, v1, v2):
    return b1*q1*(a2 - a1)*np.exp(-b1*x)*np.power(c1 + q1*np.exp(-b1*x), -1/v1 - 1)/v1 + b2*q2*(a3 - a2)*np.exp(-b2*x)*np.power(c2 + q2*np.exp(-b2*x), -1/v2 - 1)/v2

def double_glf_dx2(x, a1, a2, a3, c1, c2, b1, b2, q1, q2, v1, v2):
    sub_1 = (a2 - a1)*(-b1**2*q1**2*(-1/v1 - 1)*np.exp(-2*b1*x)*np.power(c1 + q1*np.exp(-b1*x), -1/v1 - 2)/v1 - b1**2*q1*np.exp(-b1*x)*np.power(c1 + q1*np.exp(-b1*x), -1/v1 - 1)/v1)
    sub_2 = (a3 - a2)*(-b2**2*q2**2*(-1/v2 - 1)*np.exp(-2*b2*x)*np.power(c2 + q2*np.exp(-b2*x), -1/v2 - 2)/v2 - b2**2*q2*np.exp(-b2*x)*np.power(c2 + q2*np.exp(-b2*x), -1/v2 - 1)/v2)
    return sub_1 + sub_2

def find_extrema(x, threshold = 0):
    # Finds extrema ignoring the end points
    x_sub = x[1:len(x) - 1]
    x_up = x[2::]
    x_down = x[0:len(x) - 2]
    x_1 = x_sub - x_up
    x_2 = x_sub - x_down
    x_maxima = np.argwhere((x_1 > 0) & (x_2 > 0)).flatten()
    x_minima = np.argwhere((x_1 < 0) & (x_2 < 0)).flatten()

    x_minima = x_minima[np.argwhere((x_sub[x_minima] >= threshold)).flatten()]
    x_maxima = x_maxima[np.argwhere((x_sub[x_maxima] >= threshold)).flatten()]
    return x_maxima + 1, x_minima + 1

# Regression based heuristics
def double_GLF_heuristic(radii, simplex_measure, tri, d = 2, a_mid = .5, heuristic = 'minima', opt_method = 'ampgo', eval_parameters = ['geom', 30000], debug = False, opt_kws = dict()):
    """

    Parameters
    ----------
    radii : array of shape [n_centers, n_features]
    
    simplex_measure : array of shape [n_gaussian_samples, n_features]
        Generated by make_multivariate_gaussians

    opt_method : string, compatible with lmfit
        Determines optimization routine for regression, recommend global optimization
        such as ampgo or basinhopping
    
    heuristic : string

    eval_parameters : list of length 2
    Returns
    -------

    """
    orig_x = np.array(radii[np.argsort(radii)])
    fit_x = orig_x
    n_inflection = 0

    fit_y = np.cumsum(simplex_measure[np.argsort(radii)])/np.sum(simplex_measure)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        d_glf_model = Model(double_glf)
        d_glf_model.set_param_hint('a1', value = 0, min = 0, max = 1, vary = False)
        d_glf_model.set_param_hint('adiff1', value = a_mid, min = 0.0005, max = 1)
        d_glf_model.set_param_hint('a2', value = .5, min = 0, expr = 'adiff1 + a1')
        d_glf_model.set_param_hint('a3', value = 1, min = 0, vary = False)
        d_glf_model.set_param_hint('b1', value = 1, min = 0, max = 1000)
        d_glf_model.set_param_hint('b2', value = 1, min = 0, max = 1000)
        d_glf_model.set_param_hint('c1', value = 1, min = 0.00001, vary = False)
        d_glf_model.set_param_hint('c2', value = 1, min = 0.00001, vary = False)
        d_glf_model.set_param_hint('q1', value = 1, min = 0.00001, max = 1)
        d_glf_model.set_param_hint('q2', value = 1, min = 0.00001, max = 1)
        d_glf_model.set_param_hint('v1', value = 1, min = 0.00001, max = 100)
        d_glf_model.set_param_hint('v2', value = 1, min = 0.00001, max = 100)
    
        d_glf_result = d_glf_model.fit(fit_y, x = fit_x, method = opt_method, nan_policy = 'propagate', fit_kws = opt_kws)
    
    if eval_parameters[0] == 'geom': 
        x_eval = np.geomspace(np.min(radii), np.max(radii), num = eval_parameters[1])
    elif eval_parameters[0] == 'linear':
        x_eval = np.linspace(np.min(radii), np.max(radii), num = eval_parameters[1])
    
    # Determine slope and concavity for use in determining the optimal alpha value

    y_eval = double_glf(x_eval, d_glf_result.params['a1'].value, d_glf_result.params['a2'].value, \
                        d_glf_result.params['a3'].value, d_glf_result.params['c1'].value, \
                        d_glf_result.params['c2'].value, d_glf_result.params['b1'].value, \
                        d_glf_result.params['b2'].value, d_glf_result.params['q1'].value, \
                        d_glf_result.params['q2'].value, d_glf_result.params['v1'].value, \
                        d_glf_result.params['v2'].value)

    y_slope = double_glf_dx(x_eval, d_glf_result.params['a1'].value, d_glf_result.params['a2'].value, \
                        d_glf_result.params['a3'].value, d_glf_result.params['c1'].value, \
                        d_glf_result.params['c2'].value, d_glf_result.params['b1'].value, \
                        d_glf_result.params['b2'].value, d_glf_result.params['q1'].value, \
                        d_glf_result.params['q2'].value, d_glf_result.params['v1'].value, \
                        d_glf_result.params['v2'].value)

    y_con = double_glf_dx2(x_eval, d_glf_result.params['a1'].value, d_glf_result.params['a2'].value, \
                           d_glf_result.params['a3'].value, d_glf_result.params['c1'].value, \
                           d_glf_result.params['c2'].value, d_glf_result.params['b1'].value, \
                           d_glf_result.params['b2'].value, d_glf_result.params['q1'].value, \
                           d_glf_result.params['q2'].value, d_glf_result.params['v1'].value, \
                           d_glf_result.params['v2'].value)

    peaks, __ = find_peaks(y_slope, width = 2, height = .005)
    maxima, minima = find_extrema(y_slope, threshold = 0)
    inflection, __ = find_peaks(np.abs(np.gradient(np.sign(y_con))))

    #print('Number of peaks ' + str(len(peaks)))
    #print('Number of minima ' + str(len(minima)))
    #print('Number of inflection points ' + str(len(inflection)))

    single_1 = glf(x_eval, d_glf_result.params['a1'].value, d_glf_result.params['a2'].value, \
               d_glf_result.params['c1'].value, d_glf_result.params['b1'].value, \
               d_glf_result.params['q1'].value, d_glf_result.params['v1'].value)

    single_2 = glf(x_eval, 0, d_glf_result.params['a3'].value - d_glf_result.params['a2'].value, \
               d_glf_result.params['c2'].value, d_glf_result.params['b2'].value, \
               d_glf_result.params['q2'].value, d_glf_result.params['v2'].value)
    
    plt.plot(x_eval, single_1, linestyle = '--')
    plt.plot(x_eval, single_2, linestyle = '-.')
    plt.ylabel('% Convex Volume')
    plt.xlabel('Alpha')
    plt.xscale('log')
    plt.legend(['GLF 1', 'GLF 2'])
    if debug == True:
        plt.show()
    else:
        plt.close()

    plt.figure()
    plt.plot(fit_x, fit_y, linestyle = 'none', marker = '.')
    plt.plot(x_eval, y_eval, linestyle = '-.')
    plt.ylabel('% Convex Volume')
    plt.xlabel('Alpha')
    plt.xscale('log')
    plt.show()
    #d_glf_result.plot()
    plt.title(opt_method)
    plt.axhline(d_glf_result.params['a1'].value)
    plt.axhline(d_glf_result.params['a3'].value - d_glf_result.params['a2'].value)
    plt.axhline(d_glf_result.params['a3'].value)
    plt.xscale('log')
    
    for i in peaks:
        plt.axvline(x_eval[i], color = 'r', linestyle = '--')

    for i in minima:
        for k in range(len(peaks) - 1):
            if i > peaks[k] and i < peaks[k + 1]:
                plt.axvline(x_eval[i], color = 'r', linestyle = '--')

    for i in inflection:
        plt.axvline(x_eval[i], color = 'k', linestyle = '--')

    if debug == True:
        plt.show()
    else:
        plt.close()

    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(fit_x, fit_y)
    axs[0].plot(x_eval, y_eval)
    axs[1].plot(x_eval, y_slope)
    axs[2].plot(x_eval, y_con)
    axs[0].set_xlabel('Alpha')
    axs[1].set_ylabel('Volume')
    axs[1].set_ylabel('First Derivative')
    axs[1].set_ylim(0, 1.5*np.min(y_slope[peaks]))
    axs[2].set_ylabel('Second Derivative')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')
    
    for i in peaks:
        axs[0].axvline(x_eval[i], color = 'r', linestyle = '--')
        axs[1].axvline(x_eval[i], color = 'r', linestyle = '--')
        
    for i in minima:
        for k in range(len(peaks) - 1):
            if i > peaks[k] and i < peaks[k + 1]:
                axs[0].axvline(x_eval[i], color = 'r', linestyle = '--')
                axs[1].axvline(x_eval[i], color = 'r', linestyle = '--')
                
    for i in inflection:
        axs[1].axvline(x_eval[i], color = 'k', linestyle = '--')
        axs[0].axvline(x_eval[i], color = 'k', linestyle = '--')
    if debug == True:
        plt.show()
    else:
        plt.close()

    optimal_alpha = x_eval[-1] + 1
    if len(peaks) == 1:
        message = 'Convex: Only one region of maximal slope'
        if debug:
            print(message)
        flag = False
        optimal_alpha = x_eval[-1] + 1
    
    else:
        if heuristic == 'minima':
            n_inflection = 2
            for i in minima:
                for k in range(len(peaks) - 1):
                    if i > peaks[k] and i < peaks[k + 1]:
                        #print(x_eval[i])
                        alpha_boolean = alpha_hull.alpha_shape(tri, radii, x_eval[i])
                        edges, vertices = alpha_hull.collect_alpha_edges(tri, alpha_boolean, d = d)
                        G_alpha = network_from_collection(edges, vertices)
                        n_components = nx.number_connected_components(G_alpha)

                        if n_components > 1:
                            message = 'Convex: Slope minima is disjoint'
                            if debug:
                                print(message)
                            flag = False
                            optimal_alpha = x_eval[-1] + 1

                        elif len(vertices) != len(tri.points):
                            message = 'Convex: Slope minima only contains ' + str(100*len(vertices)/len(tri.points)) + ' percent of samples'
                            if debug:
                                print(message)
                            flag = False
                            optimal_alpha = x_eval[-1] + 1
                        
                        else:
                            message = 'Concave: Slope minima contains all samples'
                            if debug:
                                print(message)
                            flag = True
                            optimal_alpha = x_eval[i]
        
        elif heuristic == 'inflection':
            counter = 1
            for i in inflection[1::]:
                #print(x_eval[i])
                counter += 1
                alpha_boolean = alpha_hull.alpha_shape(tri, radii, x_eval[i])
                edges, vertices = alpha_hull.collect_alpha_edges(tri, alpha_boolean, d = d)
                G_alpha = network_from_collection(edges, vertices)
                n_components = nx.number_connected_components(G_alpha)

                if counter >= 4:
                    message = 'Convex: An inflection point exceeding the third contains all the points and is most likely noise.'
                    if debug:
                        print(message)
                    flag = False
                    optimal_alpha = x_eval[-1] + 1
                    break

                elif n_components > 1:
                    message = 'Convex: The inflection point, (' + str(counter) + '), at ' + str(i) + ' is disjoint'
                    if debug:
                        print(message)
                    flag = False
                    optimal_alpha = x_eval[-1] + 1

                elif len(vertices) == len(tri.points):
                    message = 'Concave: The inflection point, (' + str(counter) + '), at ' + str(i) + ' contains ' + str(100*len(vertices)/len(tri.points)) + ' percent of samples'
                    flag = True
                    if debug:
                        print(message)
                    optimal_alpha = x_eval[i]
                    n_inflection = counter
                    break
                
                elif counter == len(inflection):
                    message = 'Convex: None of the inflection points contained all samples'
                    flag = False
                    if debug:
                        print(message)
                    optimal_alpha = x_eval[-1] + 1
                
    #print(message)
    return optimal_alpha, d_glf_result, n_inflection, flag, message
                

