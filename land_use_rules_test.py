import numpy as np
from neighborhood_utils import neighborhood_effect
from accessibility_utils import find_nearest_road, compute_accessibility
from stochastic_functions import stochastic_perturbation
from suitability_utils import suitability, load_suitability
from land_demand_utils import has_demand  

LAND_UNDEVELOPED = 0
LAND_LOWPRICE = 1
LAND_HIGHPRICE = 2
LAND_FIXED = 3
LAND_ROAD = 4

stochastic_x_value = 2

# Kernels for neighborhood effect
kernel_inner = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])

kernel_outer = np.array([[1, 1, 1, 1, 1],
                         [1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1],
                         [1, 1, 1, 1, 1]])

radius = 200
Ni_low = 1
Ni_high = 1

#def land_use_transformation(map, suitability_map, accessibility_map, demand_map):
#    load_suitability(suitability_map)
#    
#    new_map = map.copy()
#    undeveloped_list = []
#    lowprice_list = []
#    
#    Vi_matrix = np.zeros(map.shape)
#    
#    low_inner, high_inner = neighborhood_effect(map, kernel_inner)
#    low_outer, high_outer = neighborhood_effect(map, kernel_outer)
#
#    for i in range(map.shape[0]):
#        for j in range(map.shape[1]):
#            if map[i, j] == LAND_FIXED or suitability(i, j) == 0:
#                continue
#            Ni_low_inner = low_inner[i, j]
#            Ni_high_inner = high_inner[i, j]
#            Ni_low_outer = low_outer[i, j]
#            Ni_high_outer = high_outer[i, j]
#
#            Ni_inner = 1 - 1 / (1 + (Ni_low * Ni_low_inner + Ni_high * Ni_high_inner) / 200)
#            Ni_outer = 1 - 1 / (1 + (Ni_low * Ni_low_outer + Ni_high * Ni_high_outer) / 346)
#            Ni = 1 + 2 * Ni_inner + Ni_outer
#
#            ri = stochastic_perturbation(stochastic_x_value)
#            Ri = accessibility_map[i, j]
#            Si = suitability(i, j)
#            
#            D = demand_map[i, j]
#
#            Vi = ri * Ni * Ri * Si * D 
#            
#            Vi_matrix[i, j] = Vi
#            
#            if map[i, j] == LAND_UNDEVELOPED:
#                undeveloped_list.append((Vi, i, j))
#            elif map[i, j] == LAND_LOWPRICE:
#                lowprice_list.append((Vi, i, j))
#    
#    undeveloped_sorted = sorted(undeveloped_list, reverse=True, key=lambda x: x[0])
#    lowprice_sorted = sorted(lowprice_list, reverse=True, key=lambda x: x[0])
#
#    cutoff_5_percent = int(0.00025 * len(undeveloped_sorted))  # Reduced the percentage to slow down growth
#    #cutoff_30_percent = int(0.005 * len(undeveloped_sorted))  # Reduced the percentage to slow down growth 7.30change_to result = 1539  16225
#    cutoff_30_percent = int(0.001 * len(undeveloped_sorted))  # Reduced the percentage to slow down growth result = 3210 2209
#
#
#
#    for index, (_, i, j) in enumerate(undeveloped_sorted):
#        if index < cutoff_5_percent:
#            new_map[i, j] = LAND_HIGHPRICE
#        elif index < cutoff_30_percent:
#            new_map[i, j] = LAND_LOWPRICE
#
#
#    return new_map, Vi_matrix

def land_use_transformation(map, suitability_map, accessibility_map, demand_map):
    load_suitability(suitability_map)
    
    new_map = map.copy()
    undeveloped_list = []
    lowprice_list = []
    
    Vi_matrix = np.zeros(map.shape)
    vi_values_within_cutoff = []
    
    low_inner, high_inner = neighborhood_effect(map, kernel_inner)
    low_outer, high_outer = neighborhood_effect(map, kernel_outer)

    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            #if map[i, j] == LAND_FIXED or map[i, j] == LAND_LOWPRICE  or  map[i, j] == LAND_HIGHPRICE or  map[i, j] == LAND_ROAD or suitability(i, j) == 0 or demand_map[i, j] <= 0:
            #    continue
            if map[i, j] == LAND_UNDEVELOPED and demand_map[i, j] > 0 :
                Ni_low_inner = low_inner[i, j]
                Ni_high_inner = high_inner[i, j]
                Ni_low_outer = low_outer[i, j]
                Ni_high_outer = high_outer[i, j]

                Ni_inner = 1 - 1 / (1 + (Ni_low * Ni_low_inner + Ni_high * Ni_high_inner) / 200)
                Ni_outer = 1 - 1 / (1 + (Ni_low * Ni_low_outer + Ni_high * Ni_high_outer) / 346)
                Ni = 1 + 2 * Ni_inner + Ni_outer

                ri = stochastic_perturbation(stochastic_x_value)
                Ri = accessibility_map[i, j]
                Si = suitability(i, j)

                D = demand_map[i, j]

                Vi = ri * Ni * Ri * Si * D 

                Vi_matrix[i, j] = Vi

                if map[i, j] == LAND_UNDEVELOPED:
                    undeveloped_list.append((Vi, i, j))
                elif map[i, j] == LAND_LOWPRICE:
                    lowprice_list.append((Vi, i, j))
    
    # Only sort and use those points where demand_map > 0
    undeveloped_sorted = sorted(undeveloped_list, reverse=True, key=lambda x: x[0])
    lowprice_sorted = sorted(lowprice_list, reverse=True, key=lambda x: x[0])

    cutoff_5_percent = int(0.000025 * len(undeveloped_sorted))  # Reduced the percentage to slow down growth
    cutoff_30_percent = int(0.1 * len(undeveloped_sorted))   # Reduced the percentage to slow down growth result = 3210 2209
    print(f"cutoff_30_percent : {cutoff_30_percent}")
    for index, (_, i, j) in enumerate(undeveloped_sorted):
        if index < cutoff_5_percent:
            new_map[i, j] = LAND_HIGHPRICE
        elif index < cutoff_30_percent:
            new_map[i, j] = LAND_LOWPRICE
            vi_values_within_cutoff.append(Vi_matrix[i][j])
        elif index == cutoff_30_percent:
            print(Vi_matrix[i][j])

    vi_min = np.min(Vi_matrix)
    vi_max = np.max(Vi_matrix)
    vi_mean = np.mean(Vi_matrix)
    vi_std = np.std(Vi_matrix)
    Vi_0 = np.sum(Vi_matrix > 0)
    print(f"Number of elements in Vi_matrix greater than 0: {Vi_0}")
    print(f"Summary of Vi_matrix:\nMin: {vi_min}\nMax: {vi_max}\nMean: {vi_mean}\nStd: {vi_std}")

    return new_map, Vi_matrix