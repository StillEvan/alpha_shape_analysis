import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import distance, Delaunay
import pandas as pd
import sys
import warnings
import time as time
from alpha_shape_analysis import alpha_hull, alpha_heuristics, simplex_property_determination

# User parameters
n_samples = 2000
random_modifier = 10
noise_factor = .1
#random_state = 10
d = 2

savepath = 'results/accuracy-loop.csv'

n_runs = 100
kws = dict(totaliter=25, disp=0, eps1=.5, eps2=.0001)
subsample_list = list(range(50, 1501, 50))
# End user parameters

coords, labels = datasets.make_circles(n_samples = n_samples*2, noise = noise_factor)

new_coords = coords[labels.astype(bool)]

# Initialize storage dataframe
df = pd.DataFrame(columns = ['n_points', 'Accuracy', 'Avg_AIC', 'Std_AIC', 'Avg_R2', 'Std_R2', '1 Inflection', '2 Inflection', '3 Inflection', 'Avg_Alpha', 'Std_Alpha', \
                                  'Avg_Alpha_Vol_2', 'Std_Alpha_Vol_2', 'Avg_Alpha_Vol_3', 'Std_Alpha_Vol_3', 'Avg_Convex_Vol', 'Std_Convex_Vol', 
                                  'Avg_Percent_Volume_2', 'Std_Percent_Volume_2', 'Avg_Percent_Volume_3', 'Std_Percent_Volume_3', 'Avg_Time', 'Std_Time'])
d = len(new_coords[0])
tri = Delaunay(new_coords, incremental=True)
volume, radii, __, __ = simplex_property_determination.get_simplex_properties(tri, d = d)
print('Simplex property determination done')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    opt_alpha, d_glf_result, R2, n_inflection, flag, message = alpha_heuristics.double_GLF_heuristic(radii, volume, tri, \
                                                 opt_method = 'ampgo', heuristic = 'inflection', d = d, opt_kws = kws, debug = False)
alpha_boolean = alpha_hull.alpha_shape(tri, radii, opt_alpha)
best_volume = np.sum(volume[alpha_boolean])
print(message)
print(best_volume)
if flag == 1:
    d_glf_result.plot()
    plt.axvline(opt_alpha, color = 'r', linestyle = '--')
    plt.xscale('log')
    alpha_plot = alpha_hull.plot_alpha_tri(tri, alpha_boolean, d = d)

    plt.show()
    
t_origin = time.time()
for i in subsample_list:
    print('New sample number', str(i))
    temp_aic = []
    temp_R2 = []
    temp_volumes = []
    temp_convexity = []
    temp_alphas = []
    temp_times = []
    temp_convex_volume = []
    temp_n_inflection = []
    for j in range(n_runs):
        print('Current Sample Number ' + str(i) + 'Iteration ' + str(j) + ' out of ' + str(n_runs))
        coords_temp = new_coords[np.random.choice(new_coords.shape[0], i, replace=False), :]
        d = len(coords_temp[0])
        t1 = time.time()
        tri = Delaunay(coords_temp, incremental=True)
        volume, radii, __, __ = simplex_property_determination.get_simplex_properties(tri, d = d)
        #print('Simplex property determination done')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt_alpha, d_glf_result, R2, n_inflection, flag, message = alpha_heuristics.double_GLF_heuristic(radii, volume, \
                                                                       tri, opt_method = 'ampgo', heuristic = 'inflection', d = d, opt_kws = kws, debug = False)

                #print(d_glf_result.fit_report())
        temp_aic.append(d_glf_result.aic)
        temp_R2.append(R2)
        temp_convexity.append(flag)
        temp_alphas.append(opt_alpha)
        temp_n_inflection.append(n_inflection)
        #print(message)
        alpha_boolean = alpha_hull.alpha_shape(tri, radii, opt_alpha)

        temp_volumes.append(np.sum(volume[alpha_boolean]))
        temp_convex_volume.append(np.sum(volume))
        if flag == 1:
            d_glf_result.plot()
            plt.axvline(opt_alpha, color = 'r', linestyle = '--')
            plt.xscale('log')
            alpha_plot = alpha_hull.plot_alpha_tri(tri, alpha_boolean, d = d)
            plt.title('Loop')
            plt.close()
        print('Elapsed Time', time.time() - t_origin, 'Time for current iteration', time.time() - t1)
        temp_times.append(time.time() - t1)
    
    temp_volumes = np.array(temp_volumes)
    temp_n_inflection = np.array(temp_n_inflection)
    temp_convex_volume = np.array(temp_convex_volume)
    temp_alphas = np.array(temp_alphas)
    temp_percent = np.divide(temp_volumes, temp_convex_volume)
    #print(temp_percent)
    #print(temp_n_inflection)
    if np.sum(temp_convexity) == 0:
        print('???')
        df2 = pd.DataFrame(np.array([i, (np.sum(temp_convexity))/n_runs, np.mean(temp_aic), np.std(temp_aic), np.mean(temp_R2), np.std(temp_R2), 0, 0, \
                       0, 0, 0, \
                       0, 0, 0, \
                       0, np.mean(temp_convex_volume[np.invert(temp_convexity)]), \
                       np.std(temp_convex_volume[np.invert(temp_convexity)]), 0, \
                       0, 0, 0, np.mean(temp_times), np.std(temp_times)]).reshape(1,23), \
                       columns = ['n_points', 'Accuracy', 'Avg_AIC', 'Std_AIC', 'Avg_R2', 'Std_R2', '1 Inflection', '2 Inflection', '3 Inflection', 'Avg_Alpha', 'Std_Alpha', \
                                  'Avg_Alpha_Vol_2', 'Std_Alpha_Vol_2', 'Avg_Alpha_Vol_3', 'Std_Alpha_Vol_3', 'Avg_Convex_Vol', 'Std_Convex_Vol', 
                                  'Avg_Percent_Volume_2', 'Std_Percent_Volume_2', 'Avg_Percent_Volume_3', 'Std_Percent_Volume_3', 'Avg_Time', 'Std_Time'])
    else:
        df2 = pd.DataFrame(np.array([i, (np.sum(temp_convexity))/n_runs, np.mean(temp_aic), np.std(temp_aic), np.mean(temp_R2), np.std(temp_R2), np.sum(temp_n_inflection == 1), np.sum(temp_n_inflection == 2),\
                       np.sum(temp_n_inflection == 3), np.mean(temp_alphas[temp_convexity]), \
                       np.std(temp_alphas[temp_convexity]), np.mean(temp_volumes[temp_n_inflection == 2]), \
                       np.std(temp_volumes[temp_n_inflection == 2]), np.mean(temp_volumes[temp_n_inflection == 3]), \
                       np.std(temp_volumes[temp_n_inflection == 3]), np.mean(temp_convex_volume[np.invert(temp_convexity)]), \
                       np.std(temp_convex_volume[np.invert(temp_convexity)]), np.mean(temp_percent[temp_n_inflection == 2]), \
                       np.std(temp_percent[temp_n_inflection == 2]), np.mean(temp_percent[temp_n_inflection == 3]), \
                       np.std(temp_percent[temp_n_inflection == 3]), np.mean(temp_times), np.std(temp_times)]).reshape(1,23), \
                       columns = ['n_points', 'Accuracy', 'Avg_AIC', 'Std_AIC', 'Avg_R2', 'Std_R2', '1 Inflection', '2 Inflection', '3 Inflection', 'Avg_Alpha', 'Std_Alpha', \
                                  'Avg_Alpha_Vol_2', 'Std_Alpha_Vol_2', 'Avg_Alpha_Vol_3', 'Std_Alpha_Vol_3', 'Avg_Convex_Vol', 'Std_Convex_Vol', 
                                  'Avg_Percent_Volume_2', 'Std_Percent_Volume_2', 'Avg_Percent_Volume_3', 'Std_Percent_Volume_3', 'Avg_Time', 'Std_Time'])
    
    df = df.append(df2, ignore_index = True)

    df.to_csv(savepath)