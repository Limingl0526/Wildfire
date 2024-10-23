import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from land_use_rules_test import land_use_transformation, LAND_UNDEVELOPED, LAND_LOWPRICE, LAND_HIGHPRICE, LAND_FIXED, LAND_ROAD
from random_walk import create_random_walk_demand_map
from Wildfire_Risk_Map import set_radius_6_to_2
from burned_cells import calculate_burned_cells
from Tolerance import Tolerance_maps
from bayes_opt import BayesianOptimization
import logging

def load_maps():
    size = (640, 873)

    suitability_map = pd.read_csv("/home/limingl/wildfire/Code/Suitablity.csv", header=None).values
    initial_map = pd.read_csv("/home/limingl/wildfire/Initial_final.csv", header=None).values
    initial_population_map = pd.read_csv("/home/limingl/wildfire/Population_2000.csv", header=None).values
    initial_map[initial_map == 5] = 4

    matrix1 = pd.read_csv("/home/limingl/wildfire/Code/R_highway.csv", header=None).to_numpy()
    matrix2 = pd.read_csv("/home/limingl/wildfire/Code/R_coastline.csv", header=None).to_numpy()
    accessibility_map = (1 + matrix1) * (1 + 0.2 * matrix2)

    biome_map = pd.read_csv("/home/limingl/wildfire/Code/Biome.csv", header=None).to_numpy()
    vegetation_map = pd.read_csv("/home/limingl/wildfire/Code/Vegetation.csv", header=None).to_numpy()
    risk_map = pd.read_csv("/home/limingl/wildfire/Code/Output_10/WUI_Per_matrix.csv", header=None).to_numpy()

    return size, suitability_map, initial_map, initial_population_map, accessibility_map, biome_map, vegetation_map, risk_map


def normalize(matrix, min_value=0.01, max_value=1.0):
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)

    if matrix_max == matrix_min:
        return np.full_like(matrix, min_value)

    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min) * (max_value - min_value) + min_value
    return normalized_matrix


def simulate_land_use(map, population_map, risk_map, steps, A, B, C, D, size=(640, 873)):
    num_different_positions_list = []

    for step in range(steps):
        burned_map_folder = "/home/limingl/wildfire/Code/Output_10/Fire_map/"
        bill_file = "/home/limingl/wildfire/Code/Output_10/Forest_Education.xlsx"
        relief_map_folder = "/home/limingl/wildfire/Code/Output_10/Relief_map/"
        Historical_map_folder = "/home/limingl/wildfire/Code/Output_10/Historical_Land_Use/"
        
        Tolerance_map = Tolerance_maps(burned_map_folder, bill_file, relief_map_folder, A, B, C, D, step)
        if Tolerance_map is None:
            raise ValueError("Tolerance_map cannot be None.")

        growth_rate_map = create_random_walk_demand_map(size, mean=2.68, std=1.86)
        demand_map = population_map * (1 + growth_rate_map) * Tolerance_map
        demand_map = normalize(demand_map)

        risk_map[set_radius_6_to_2(risk_map) == 1] = 0
        new_map = land_use_transformation(map, suitability_map, accessibility_map, demand_map)
        new_map[map == LAND_ROAD] = LAND_ROAD

        burned_map_files = sorted(os.listdir(burned_map_folder))
        burned_map_path = os.path.join(burned_map_folder, burned_map_files[step])
        Wildfire_risk = pd.read_csv(burned_map_path, header=None).values

        new_map, burned_map = calculate_burned_cells(Wildfire_risk, new_map)
        population_map = np.where(burned_map, 0.01, demand_map)
        map = new_map

        Historical_map_files = sorted(os.listdir(Historical_map_folder))
        Historical_map_path = os.path.join(Historical_map_folder, Historical_map_files[step])
        Historical_land_use = pd.read_csv(Historical_map_path, header=None).values

        differences = ((Historical_land_use == 1) | (Historical_land_use == 2)) & (new_map == 0)
        num_different_positions = np.sum(differences)
        num_different_positions_list.append(num_different_positions)

    return new_map, num_different_positions_list


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def black_box_function(A, B, C, D):
    try:
        #if A + B > C + D and A > B and C < D:
        final_map, num_different_positions_list = simulate_land_use(initial_map, initial_population_map, risk_map, steps, A, B, C, D)
        num_different_positions = num_different_positions_list[-1]
        if np.isnan(num_different_positions) or np.isinf(num_different_positions):
            logging.warning(f"Invalid number of different positions: {num_different_positions} for A={A}, B={B}, C={C}, D={D}")
            return float('-10000')
        return -num_different_positions  # Negative because BayesianOptimization maximizes by default
        #return float('-inf')
    except Exception as e:
        logging.error(f"Error in black_box_function: {e}", exc_info=True)
        return float('-inf')

def run_bayesian_optimization(output_csv='bayesian_optimization_results.csv'):
    optimizer = BayesianOptimization(
        f=black_box_function, 
        pbounds={'A': (1, 10), 'B': (1, 10), 'C': (1, 10), 'D': (1, 10)}, 
        random_state=2
    )

    logging.info("Starting Bayesian Optimization")
    
    optimizer.maximize(init_points=9, n_iter=90)

    if not os.path.isfile(output_csv):
        with open(output_csv, 'w') as f:
            f.write("iteration,target,A,B,C,D,num_different_positions_list\n")

    for i, res in enumerate(optimizer.res):
        params = res['params']
        A, B, C, D = params['A'], params['B'], params['C'], params['D']
        
        try:
            final_map, num_different_positions_list = simulate_land_use(initial_map, initial_population_map, risk_map, steps, A, B, C, D)
            target = res['target']
            num_different_positions_list_str = ','.join(map(str, num_different_positions_list))

            if not np.isnan(target) and not np.isinf(target):
                with open(output_csv, 'a') as f:
                    f.write(f"{i+1},{target},{A},{B},{C},{D},{num_different_positions_list_str}\n")
            else:
                logging.warning(f"Skipping invalid target value for params A={A}, B={B}, C={C}, D={D}")
        except Exception as e:
            logging.error(f"Error during optimization iteration {i+1}: {e}", exc_info=True)
            with open(output_csv, 'a') as f:
                f.write(f"{i+1},N/A,{A},{B},{C},{D},Error during iteration\n")

if __name__ == "__main__":
    try:
        size, suitability_map, initial_map, initial_population_map, accessibility_map, biome_map, vegetation_map, risk_map = load_maps()
        relief_cum_map = np.ones((640, 873))
        steps = 20
        run_bayesian_optimization()
    except Exception as e:
        logging.critical(f"Critical error in the main execution: {e}", exc_info=True)