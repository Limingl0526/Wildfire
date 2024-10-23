import pandas as pd
import numpy as np

def process_matrices(matrix1, matrix2, matrix3, mask_matrix):
    # 按掩码矩阵进行元素相乘
    result_matrix = np.where(mask_matrix != 0, matrix1 * matrix2 * (matrix3)**0.05, 0)

    # 计算结果矩阵的最小值和最大值
    min_val = np.min(result_matrix)
    max_val = np.max(result_matrix)

    # 归一化处理，避免除以零错误
    if min_val != max_val:
        result_matrix_normalized = (result_matrix - min_val) / (max_val - min_val)
    else:
        result_matrix_normalized = np.zeros_like(result_matrix)

    # 将归一化后的结果矩阵转换为DataFrame
    result_df_normalized = pd.DataFrame(result_matrix_normalized)

    return result_df_normalized

