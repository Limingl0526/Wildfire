
suitability_map = None 

def load_suitability(map):
    global suitability_map
    suitability_map = map

def suitability(i, j):
    if suitability_map is None:
        raise ValueError("Suitability map has not been loaded.")
    return suitability_map[i, j]