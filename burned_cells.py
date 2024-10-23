import numpy as np
from scipy.ndimage import binary_dilation

def calculate_burned_cells(Wildfire_risk, current_map):
    # Constants
    LAND_FIXED = 3
    LAND_ROAD = 4
    LAND_UNDEVELOPED = 0
    LAND_LOWPRICE = 1
    LAND_HIGHPRICE = 2
    
    random_numbers = np.random.rand(*Wildfire_risk.shape)  # Generate uniform distribution numbers between 0 and 1

    # Initialize burned map
    burned_map = np.zeros(Wildfire_risk.shape, dtype=bool)
    
    # Determine initial burned area based on the random numbers and wildfire risk
    initial_mask = (random_numbers < Wildfire_risk)  & (current_map != LAND_FIXED) & (current_map != LAND_ROAD)
    burned_map[initial_mask] = True
    
    # Iteratively expand burned area
    #for _ in range(60):  # Assume 60 steps expansion (can be adjusted)
    #    # Identify the potential new burn area
    #    new_burn_area = binary_dilation(burned_map) & ~burned_map
    #    
    #    # Apply conditions within the new potential burn area
    #    valid_burn_area = (new_burn_area & (Wildfire_risk > 0.5) & 
    #                       (current_map != LAND_FIXED) & 
    #                       (current_map != LAND_ROAD))
    #    
    #    # Update the burned map
    #    burned_map[valid_burn_area] = True
    
    # Create a copy of the current map to update cells
    new_map = np.copy(current_map)
    new_map[burned_map] = LAND_UNDEVELOPED
    
    # Burned low price lands
    burned_low = burned_map & (current_map == LAND_LOWPRICE)
    burned_map[burned_low] = 1
    count_burned_low_cells = np.sum(burned_low)
    #print(f'Total burned low price cells count: {count_burned_low_cells}')
    
    # Burned high price lands
    burned_high = burned_map & (current_map == LAND_HIGHPRICE)
    burned_map[burned_high] = 1
    count_burned_high_cells = np.sum(burned_high)
    #print(f'Total burned high price cells count: {count_burned_high_cells}')
    
    # Count total burned cells again
    count_burned_cells = np.sum(burned_map)
    #print(f'Total burned cells count: {count_burned_cells}')

    #return new_map, burned_map.astype(int)  
    return count_burned_cells, burned_map.astype(int)
