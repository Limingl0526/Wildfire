import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from land_use_rules_test import land_use_transformation, LAND_UNDEVELOPED, LAND_LOWPRICE, LAND_HIGHPRICE, LAND_FIXED, LAND_ROAD
from random_walk import run_dla_simulation
from Wildfire_Risk_Map import set_radius_6_to_2
from Wildfire_risk import process_matrices
from burned_cells import calculate_burned_cells
from Tolerance import Tolerance_maps
import os
import matplotlib.colors as mcolors

# 模型参数初始化
size = (640, 873)
steps = 20
num_experiments = 5

# 数据读取
suitability_map = pd.read_csv("/home/limingl/wildfire/CA_model/Suitablity.csv", header=None).values
initial_map = pd.read_csv("/home/limingl/wildfire/CA_model/Initial_final.csv", header=None).values
initial_population_map = pd.read_csv("/home/limingl/wildfire/CA_model/Population_2000.csv", header=None).values

initial_population_map[initial_population_map == 0.01] = 0
initial_map[initial_map == 5] = 4

matrix1 = pd.read_csv("/home/limingl/wildfire/CA_model/R_highway.csv", header=None).to_numpy()
matrix2 = pd.read_csv("/home/limingl/wildfire/CA_model/R_coastline.csv", header=None).to_numpy()
accessibility_map = (1 + matrix1) * (1 + matrix2)

bimoe_map = pd.read_csv("/home/limingl/wildfire/Biome.csv", header=None).to_numpy()
vegetation_map = pd.read_csv("/home/limingl/wildfire/Vegetation.csv", header=None).to_numpy()
risk_map = pd.read_csv("/home/limingl/wildfire/CA_model/inner_coastline/WUI_Per_matrix.csv", header=None).to_numpy()

# 颜色映射
colors = {
    LAND_UNDEVELOPED: "#BDFFE2",
    LAND_LOWPRICE: "#FFFF71",
    LAND_HIGHPRICE: "#FFFF71",
    LAND_FIXED: "#BED2FF",
    LAND_ROAD: "#B6ABBA"
}
cmap = ListedColormap([colors[x] for x in [LAND_UNDEVELOPED, LAND_LOWPRICE, LAND_HIGHPRICE, LAND_FIXED, LAND_ROAD]])

colors = [(1, 1, 1), (1, 0, 0)]  # 从白到红
n_bins = 100  # 颜色梯度的级别数
cmap_name = 'white_to_red'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def simulate_land_use(initial_map, initial_population_map, risk_map, steps, size=(640, 873)):
    map = initial_map.copy()
    population_map = initial_population_map.copy()
    
    for step in range(steps):
        
        # 定义路径
        burned_map_folder = '/home/limingl/wildfire/CA_model/Output_10/Fire_map'
        bill_file = '/home/limingl/wildfire/CA_model/Output_10/Forest_Education.xlsx'
        relief_map_folder = '/home/limingl/wildfire/CA_model/Output_10/Relief_map'
        Historical_map_folder = '/home/limingl/wildfire/CA_model/Output_10/Historical_Land_Use'
        
        A, B, C, D = 8.212, 7.818, 1.749, 2.048
        Tolerance_map = Tolerance_maps(burned_map_folder, bill_file, relief_map_folder, A, B, C, D, step)

        pop_new = run_dla_simulation(population_map, 5)
        demand_map = pop_new * Tolerance_map
        
        mask_compare = (risk_map == 1) | (risk_map == 2)
        mask_nonzero_new_map = map != 0
        risk_map[mask_compare & mask_nonzero_new_map] = 0
        risk_map = set_radius_6_to_2(risk_map)

        new_map, Vi_matrix = land_use_transformation(map, suitability_map, accessibility_map, demand_map)
        new_map[map == LAND_ROAD] = LAND_ROAD

        # 加载 wildfire risk 地图
        burned_map_files = sorted([f for f in os.listdir(burned_map_folder) if os.path.isfile(os.path.join(burned_map_folder, f))])
        burned_map_file = burned_map_files[step]
        burned_map_path = os.path.join(burned_map_folder, burned_map_file)
        Wildfire_risk = pd.read_csv(burned_map_path, header=None).values
        count_burned_cells, burned_map = calculate_burned_cells(Wildfire_risk, new_map)

        population_map = pop_new 
        random_map = np.random.rand(*burned_map.shape) < 0.5
        burned_map_bool = burned_map.astype(bool)
        to_modify = burned_map_bool & random_map
        population_map[to_modify] = 0
        
        map = new_map

    return map

# 创建增长矩阵
growth_matrix = np.zeros(size)

for experiment in range(num_experiments):
    np.random.seed(experiment)  # 设置随机种子以确保每次实验的可重复性
    final_map = simulate_land_use(initial_map, initial_population_map, risk_map, steps)
    
    # 保存每次实验的最终结果
    output_file = f"/home/limingl/wildfire/CA_model/Output_811/Experiment_{experiment + 1}.csv"
    pd.DataFrame(final_map).to_csv(output_file, header=None, index=False)

    # 计算增长矩阵
    growth_matrix[((initial_map == 0) & ((final_map == 1) | (final_map == 2)))] += 1

# 保存增长矩阵
growth_output_file = "/home/limingl/wildfire/CA_model/Output_811/Growth_Matrix.csv"
pd.DataFrame(growth_matrix).to_csv(growth_output_file, header=None, index=False)