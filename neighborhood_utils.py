
from scipy.signal import convolve2d

def neighborhood_effect(map, kernel):
    lowprice_neighbors = convolve2d((map == 1).astype(int), kernel, mode='same', boundary='wrap')
    highprice_neighbors = convolve2d((map == 2).astype(int), kernel, mode='same', boundary='wrap')
    return lowprice_neighbors, highprice_neighbors