#import os
#import pandas as pd
#import numpy as np
#
## 全局变量，用于累加 relief map
#relief_cum_map = np.ones((640, 873))
#
#def Tolerance_maps(burned_map_folder, bill_file, relief_map_folder, A, B, C, D, step):
#    try:
#        # 声明 global relief_cum_map，以便在函数内部更改全局变量
#        global relief_cum_map
#
#        # Step 对应的文件索引
#        file_index = step
#        
#        # 获取 burned map 文件夹中的所有文件
#        burned_map_files = sorted(os.listdir(burned_map_folder))
#        if file_index >= len(burned_map_files):
#            print(f"No burned map file found for step {step}.")
#            return
#        burned_map_file = burned_map_files[file_index]
#        burned_map_path = os.path.join(burned_map_folder, burned_map_file)
#        
#        # 获取 relief map 文件夹中的所有文件
#        relief_map_files = sorted(os.listdir(relief_map_folder))
#        if file_index >= len(relief_map_files):
#            print(f"No relief map file found for step {step}.")
#            return
#        relief_map_file = relief_map_files[file_index]
#        relief_map_path = os.path.join(relief_map_folder, relief_map_file)
#        
#        # 读取 burned map 和 relief map 文件，并将它们转换为浮点数类型
#        burned_map = pd.read_csv(burned_map_path, header=None).values.astype(np.float64)
#        relief_map = pd.read_csv(relief_map_path, header=None).values.astype(np.float64)
#
#        # 检查两个矩阵大小是否相等
#        if burned_map.shape != relief_map.shape:
#            raise ValueError("Burned map and relief map must have the same dimensions.")
#        
#        # 累加 relief_map 到 relief_cum_map
#        if relief_cum_map is None:
#            relief_cum_map = relief_map
#        else:
#            relief_cum_map += relief_map
#        
#        # 读取 bill 文件（这里假设 bill 文件仍然是一个 Excel 文件）
#        bill = pd.read_excel(bill_file, header=None, engine='openpyxl')
#        if file_index >= len(bill):
#            print(f"No bill data found for step {step}.")
#            return
#        forest = bill.iloc[file_index, 0]
#        education = bill.iloc[file_index, 1]
#        
#        # 进行计算：A * relief_cum_map - D * burned_map + B * forest - C * education
#        #result = A * relief_cum_map - D * burned_map + B * forest - C * education
#        #relief_cum_map_safe = np.where(relief_cum_map > 0, relief_cum_map, 1e-3)
#        result = np.log(A * relief_cum_map + B * forest)  - D * burned_map - C * education
#        
#        # 读取 initial_population_map 并将其转换为浮点数类型
#        initial_population_map = pd.read_csv("/home/limingl/wildfire/CA_model/Population_2000.csv", header=None).values.astype(np.float64)
#        result[initial_population_map == 0] = 0
#        
#        return result
#
#    except Exception as e:
#        print(f"An error occurred: {str(e)}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def Tolerance_maps(burned_map_folder, bill_file, relief_map_folder, A, B, C, D, step):
    try:
        # 初始化只有在函数内部定义累计 relief map
        relief_cum_map = np.ones((640, 873))

        # Step 对应的文件索引
        file_index = step
        
        # 获取 burned map 文件夹中的所有文件
        burned_map_files = sorted(os.listdir(burned_map_folder))
        if file_index >= len(burned_map_files):
            print(f"No burned map file found for step {file_index}.")
            return
        burned_map_file = burned_map_files[file_index]
        burned_map_path = os.path.join(burned_map_folder, burned_map_file)
        
        # 获取 relief map 文件夹中的所有文件
        relief_map_files = sorted(os.listdir(relief_map_folder))
        if file_index >= len(relief_map_files):
            print(f"No relief map file found for step {file_index}.")
            return
        relief_map_file = relief_map_files[file_index]
        relief_map_path = os.path.join(relief_map_folder, relief_map_file)
        
        # 读取 burned map 和 relief map 文件，并将它们转换为浮点数类型
        burned_map = pd.read_csv(burned_map_path, header=None).values.astype(np.float64)
        relief_map = pd.read_csv(relief_map_path, header=None).values.astype(np.float64)

        # 检查两个矩阵大小是否相等
        if burned_map.shape != relief_map.shape:
            raise ValueError("Burned map and relief map must have the same dimensions.")
        
        # 累加 relief_map 到 relief_cum_map
        relief_cum_map += relief_map
        
        # 读取 bill 文件（这里假设 bill 文件仍然是一个 Excel 文件）
        bill = pd.read_excel(bill_file, header=None, engine='openpyxl')
        if file_index >= len(bill):
            print(f"No bill data found for step {file_index}.")
            return
        forest = bill.iloc[file_index, 0]
        education = bill.iloc[file_index, 1]
        
        # 进行计算：A * relief_cum_map - D * burned_map + B * forest - C * education
        result = np.log(A * relief_cum_map + B * forest ) - D * burned_map - C * education
        initial_population_map = pd.read_csv("/home/limingl/wildfire/CA_model/Population_2000.csv", header=None).values.astype(np.float64)
       
        # 使用全局变量 initial_population_map
        result[initial_population_map == 0] = -D

        # 将result归一化到0.01到1的范围
        min_val = np.min(result)
        max_val = np.max(result)
        normalized_result = 0.01 + (result - min_val) * (1 - 0.01) / (max_val - min_val)

        return normalized_result
        

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# 全局变量 initial_population_map



