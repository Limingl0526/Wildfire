import numpy as np
#1 forest area
#2 intermix area
#3 forest buffer area
def set_radius_6_to_2(matrix):
    rows, cols = matrix.shape
    mark_to_modify = np.zeros((rows, cols), dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                x_min, x_max = max(i - 6, 0), min(i + 6 + 1, rows)
                y_min, y_max = max(j - 6, 0), min(j + 6 + 1, cols)
                
                if np.any(matrix[x_min:x_max, y_min:y_max] == 0):
                    for x in range(x_min, x_max):
                        for y in range(y_min, y_max):
                            if np.linalg.norm([x - i, y - j]) <= 6:
                                if matrix[x, y] == 0:
                                    mark_to_modify[x, y] = True
    
    
    matrix[mark_to_modify] = 3
    
    return matrix
