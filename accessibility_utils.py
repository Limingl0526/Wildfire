#import numpy as np
#
#LAND_UNDEVELOPED = 0
#LAND_LOWPRICE = 1
#LAND_HIGHPRICE = 2
#LAND_ROAD = 4
#
#def find_nearest_road(map, i, j):
#    distances = np.full(map.shape, np.inf)
#
#    # Identify where roads are located
#    road_positions = np.argwhere(map == LAND_ROAD)
#
#    # Compute the distance from each land cell to the nearest road
#    for rp in road_positions:
#        distance = np.sqrt((rp[0] - i)**2 + (rp[1] - j)**2)
#        if distance < distances[i, j]:
#            distances[i, j] = distance
#    
#    return distances[i, j]
#
#def accessibility(map, i, j):
#    distance = find_nearest_road(map, i, j)
#    # Inverse distance metric for higher accessibility with lesser distance
#    return 1 / (1 + distance)
#import numpy as np
#
#LAND_UNDEVELOPED = 0
#LAND_LOWPRICE = 1
#LAND_HIGHPRICE = 2
#LAND_ROAD = 4
#
#def find_nearest_road(map, i, j):
#    # Identify where roads are located
#    road_positions = np.argwhere(map == LAND_ROAD)
#    
#    # Initialize minimum distance as infinity
#    min_distance = np.inf
#
#    # Compute the distance from (i, j) to each road and find the minimum distance
#    for rp in road_positions:
#        distance = np.sqrt((rp[0] - i)**2 + (rp[1] - j)**2)
#        if distance < min_distance:
#            min_distance = distance
#    
#    return min_distance
#
#def accessibility(map, i, j):
#    distance = find_nearest_road(map, i, j)
#    # Inverse distance metric for higher accessibility with lesser distance
#    return 1 / (1 + distance)

import numpy as np

LAND_UNDEVELOPED = 0
LAND_LOWPRICE = 1
LAND_HIGHPRICE = 2
LAND_ROAD = 4
def find_nearest_road(map):
    road_positions = np.argwhere(map == LAND_ROAD)
    distances = np.full(map.shape, np.inf)

    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] != LAND_ROAD:
                for rp in road_positions:
                    distance = np.sqrt((rp[0] - i)**2 + (rp[1] - j)**2)
                    if distance < distances[i, j]:
                        distances[i, j] = distance
    return distances

def compute_accessibility(distances):
    return 1 / (1 + distances)