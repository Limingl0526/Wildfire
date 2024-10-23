# 文件名: land_demand_utils.py
import numpy as np

def load_land_demand(demand_map):
    
    global land_demand
    land_demand = demand_map

def has_demand(i, j):
    
    return land_demand[i, j] == 1