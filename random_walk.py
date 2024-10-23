#import numpy as np
#import random
#import matplotlib.pyplot as plt
#
#def initialize_grid(custom_matrix):
#    return np.where(custom_matrix > 0.01, 1, 0)
#
#def is_isolated(grid, x, y):
#    size_x, size_y = grid.shape
#    neighbours = [((-1 + x) % size_x, y), ((1 + x) % size_x, y), 
#                  (x, (-1 + y) % size_y), (x, (1 + y) % size_y)]
#    
#    for nx, ny in neighbours:
#        if grid[nx, ny] == 0:
#            return False
#    return True
#
#def random_walk(grid, initial_points, custom_matrix, max_steps):
#    size_x, size_y = grid.shape
#    new_points = []
#    occupied_points = set(initial_points)
#    walked_points = []
#    
#    max_custom_value = np.max(custom_matrix)
#    min_custom_value = np.min(custom_matrix)
#    
#    while initial_points:
#        x, y = initial_points.pop(random.randint(0, len(initial_points) - 1))
#        
#        if is_isolated(grid, x, y):
#            continue
#        
#        initial_value = custom_matrix[x, y]
#
#        # Normalize the custom_matrix value to be between 0 and 1
#        normalized_value = (custom_matrix[x, y] - min_custom_value) / (max_custom_value - min_custom_value)
#        # Scale the number of steps to be between 1 and max_steps
#        #steps = int(2 + normalized_value * (max_steps - 2))
#        steps = 5
#        
#        for _ in range(steps):
#            neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#            dx, dy = random.choice(neighbours)
#            new_x = (x + dx) % size_x
#            new_y = (y + dy) % size_y
#
#            walked_points.append((new_x, new_y))
#
#            if grid[new_x, new_y] == 1:
#                grid[x, y] = 1
#                occupied_points.add((x, y))
#                new_points.append((x, y))
#                break
#
#            initial_value *= 0.01 # decrease the value by a factor of 0.01 for each step
#            custom_matrix[new_x, new_y] += initial_value
#            x, y = new_x, new_y
#
#    return grid, new_points, walked_points
#
#def dla(custom_matrix, max_steps):
#    initial_grid = initialize_grid(custom_matrix)
#    grid = initial_grid.copy()
#    initial_points = list(zip(*np.nonzero(grid)))
#    all_walked_points = []
#    
#    while initial_points:
#        grid, new_points, walked_points = random_walk(grid, initial_points, custom_matrix, max_steps)
#        initial_points = new_points
#        all_walked_points.extend(walked_points)
#    
#    return custom_matrix, initial_grid, all_walked_points
#
#def run_dla_simulation(custom_matrix, max_steps):
#    final_custom_matrix, initial_grid, walked_points = dla(custom_matrix, max_steps)
#
#    # First plot: DLA Fractal
#    plt.figure(figsize=(10, 10))
#    plt.imshow(final_custom_matrix, cmap='binary')
#    plt.title('DLA Fractal')
#    plt.axis('off')
#    plt.show()
#
#    # Check the shape of initial_grid
#    print(f"Shape of initial_grid: {initial_grid.shape}")
#
#    initial_coords = np.nonzero(initial_grid)
#    if len(initial_coords) == 2:  # Ensure it's 2D
#        initial_x, initial_y = initial_coords
#    else:
#        raise ValueError("Initial grid is not 2D")
#
#    if walked_points:
#        walked_x, walked_y = zip(*walked_points)
#    else:
#        walked_x, walked_y = [], []  # Ensure empty walked points are handled
#
#    # Determine the difference (newly walked points) between initial_grid and final_custom_matrix
#    difference_coords = np.nonzero((final_custom_matrix != 0) & (initial_grid == 0))
#    difference_x, difference_y = difference_coords
#
#    # Create figure
#    plt.figure(figsize=(8, 6))
#    plt.scatter(difference_y, difference_x, c='black', s=1)  # Using scatter plot with black dots
#    plt.title('DLA Fractal')
#    plt.gca().invert_yaxis()  # Invert y-axis if needed
#    plt.axis('off')
#    plt.show()
#
#    return final_custom_matrix

## Example usage:
#if __name__ == "__main__":
#    custom_matrix = pd.read_csv("/home/limingl/wildfire/CA_model/Population_2000.csv", header=None).values
#    custom_matrix[custom_matrix == 0.01] = 0
#    max_steps = 5  # You can set this to any desired maximum number of steps
#    final_matrix = run_dla_simulation(custom_matrix, max_steps)
#    print("Grid values sum should be approximately equal to initial sum(with some error margin):")
#    print(final_matrix.sum())
#import numpy as np
#import random
#import matplotlib.pyplot as plt
#
#def initialize_grid(custom_matrix):
#    return np.where(custom_matrix > 0.01, 1, 0)
#
#def is_isolated(grid, x, y):
#    size_x, size_y = grid.shape
#    neighbours = [((-1 + x) % size_x, y), ((1 + x) % size_x, y), 
#                  (x, (-1 + y) % size_y), (x, (1 + y) % size_y)]
#    
#    for nx, ny in neighbours:
#        if grid[nx, ny] == 0:
#            return False
#    return True
#
#def random_walk(grid, initial_points, custom_matrix, max_steps):
#    size_x, size_y = grid.shape
#    new_points = []
#    occupied_points = set(initial_points)
#    walked_points = []
#    
#    max_custom_value = np.max(custom_matrix)
#    min_custom_value = np.min(custom_matrix)
#    
#    while initial_points:
#        x, y = initial_points.pop(random.randint(0, len(initial_points) - 1))
#        
#        if is_isolated(grid, x, y):
#            continue
#        
#        initial_value = custom_matrix[x, y]
#
#        # Normalize the custom_matrix value to be between 0 and 1
#        normalized_value = (custom_matrix[x, y] - min_custom_value) / (max_custom_value - min_custom_value)
#        # Scale the number of steps to be between 1 and max_steps
#        #steps = int(2 + normalized_value * (max_steps - 2))
#        steps = 1
#        
#        for _ in range(steps):
#            neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#            dx, dy = random.choice(neighbours)
#            new_x = (x + dx) % size_x
#            new_y = (y + dy) % size_y
#
#            walked_points.append((new_x, new_y))
#
#            if grid[new_x, new_y] == 1:
#                grid[x, y] = 1
#                occupied_points.add((x, y))
#                new_points.append((x, y))
#                break
#
#            initial_value *= 0.01 # decrease the value by a factor of 0.01 for each step
#            custom_matrix[new_x, new_y] += initial_value
#            x, y = new_x, new_y
#
#    return grid, new_points, walked_points
#
#def dla(custom_matrix, max_steps):
#    initial_grid = initialize_grid(custom_matrix)
#    grid = initial_grid.copy()
#    initial_points = list(zip(*np.nonzero(grid)))
#    all_walked_points = []
#    
#    while initial_points:
#        grid, new_points, walked_points = random_walk(grid, initial_points, custom_matrix, max_steps)
#        initial_points = new_points
#        all_walked_points.extend(walked_points)
#    
#    return custom_matrix, initial_grid, all_walked_points
#
#def run_dla_simulation(custom_matrix, max_steps):
#    final_custom_matrix, initial_grid, walked_points = dla(custom_matrix, max_steps)
#
#    # First plot: DLA Fractal
#    plt.figure(figsize=(10, 10))
#    plt.imshow(final_custom_matrix, cmap='binary')
#    plt.title('DLA Fractal')
#    plt.axis('off')
#    plt.show()
#
#    # Check the shape of initial_grid
#    print(f"Shape of initial_grid: {initial_grid.shape}")
#
#    initial_coords = np.nonzero(initial_grid)
#    if len(initial_coords) == 2:  # Ensure it's 2D
#        initial_x, initial_y = initial_coords
#    else:
#        raise ValueError("Initial grid is not 2D")
#
#    if walked_points:
#        walked_x, walked_y = zip(*walked_points)
#    else:
#        walked_x, walked_y = [], []  # Ensure empty walked points are handled
#
#    # Determine the difference (newly walked points) between initial_grid and final_custom_matrix
#    difference_coords = np.nonzero((final_custom_matrix != 0) & (initial_grid == 0))
#    difference_x, difference_y = difference_coords
#
#    # Create figure to plot both initial and new points
#    plt.figure(figsize=(8, 6))
#
#    # Plot initial points in gray
#    plt.scatter(initial_y, initial_x, c='gray', s=10, label='Initial Points')
#
#    # Plot new points in red
#    plt.scatter(difference_y, difference_x, c='red', s=10, label='New Points')
#
#    plt.title('DLA Fractal')
#    plt.gca().invert_yaxis()  # Invert y-axis if needed
#    plt.axis('off')
#    plt.legend()
#    plt.show()
#
#    return final_custom_matrix

#import numpy as np
#import random
#import matplotlib.pyplot as plt
#
#def initialize_grid(custom_matrix):
#    return np.where(custom_matrix > 0.01, 1, 0)
#
#def is_isolated(grid, x, y):
#    size_x, size_y = grid.shape
#    neighbours = [((-1 + x) % size_x, y), ((1 + x) % size_x, y), 
#                  (x, (-1 + y) % size_y), (x, (1 + y) % size_y),
#                  ((-1 + x) % size_x, (-1 + y) % size_y), ((-1 + x) % size_x, (1 + y) % size_y),
#                  ((1 + x) % size_x, (-1 + y) % size_y), ((1 + x) % size_x, (1 + y) % size_y)]
#    
#    for nx, ny in neighbours:
#        if grid[nx, ny] == 0:
#            return False
#    return True
#
#def random_walk(grid, initial_points, custom_matrix, max_steps, radius_range=3):
#    size_x, size_y = grid.shape
#    new_points = []
#    walked_points = []
#    
#    max_custom_value = np.max(custom_matrix)
#    min_custom_value = np.min(custom_matrix)
#    
#    # 根据初始值大小对初始点排序
#    initial_points = sorted(initial_points, key=lambda point: custom_matrix[point[0], point[1]], reverse=True)
#    
#    while initial_points:
#        x, y = initial_points.pop(0)  # 取出初始值最大的点
#        
#        if is_isolated(grid, x, y):
#            continue
#        
#        initial_value = custom_matrix[x, y]
#
#        # Normalize the custom_matrix value to be between 0 and 1
#        normalized_value = (custom_matrix[x, y] - min_custom_value) / (max_custom_value - min_custom_value)
#        # Scale the number of steps to be between 1 and max_steps
#        steps = int(0 + normalized_value * (max_steps - 0))
#
#        for _ in range(steps):
#            if random.random() < 0.5:  # 50% 随机在周围一两格内扩展
#                neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
#                                (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 八个方向的邻居点
#                dx, dy = random.choice(neighbours)
#            else:  # 50% 随机在一定范围内探索
#                angle = 2 * np.pi * random.random()
#                radius = random.uniform(1, radius_range)  # 半径范围内随机选择一个点
#                dx, dy = int(radius * np.cos(angle)), int(radius * np.sin(angle))
#
#            new_x = (x + dx) % size_x
#            new_y = (y + dy) % size_y
#            
#            walked_points.append((new_x, new_y))
#
#            if grid[new_x, new_y] == 1:
#                grid[x, y] = 1
#                new_points.append((x, y))
#                break
#
#            initial_value *= 0.01  # decrease the value by a factor of 0.01 for each step
#            custom_matrix[new_x, new_y] += initial_value
#            x, y = new_x, new_y
#
#    return grid, new_points, walked_points
#
#def dla(custom_matrix, max_steps, radius_range=3):
#    initial_grid = initialize_grid(custom_matrix)
#    grid = initial_grid.copy()
#    initial_points = list(zip(*np.nonzero(grid)))
#    all_walked_points = []
#    
#    while initial_points:
#        grid, new_points, walked_points = random_walk(grid, initial_points, custom_matrix, max_steps, radius_range)
#        initial_points = new_points
#        all_walked_points.extend(walked_points)
#    
#    return custom_matrix, initial_grid, all_walked_points
#
#def run_dla_simulation(custom_matrix, max_steps, radius_range=5):
#    final_custom_matrix, initial_grid, walked_points = dla(custom_matrix, max_steps, radius_range)
#
#    # First plot: DLA Fractal
#    plt.figure(figsize=(10, 10))
#    plt.imshow(final_custom_matrix, cmap='binary')
#    plt.title('DLA Fractal')
#    plt.axis('off')
#    plt.show()
#
#    # Check the shape of initial_grid
#    print(f"Shape of initial_grid: {initial_grid.shape}")
#
#    initial_coords = np.nonzero(initial_grid)
#    if len(initial_coords) == 2:  # Ensure it's 2D
#        initial_x, initial_y = initial_coords
#    else:
#        raise ValueError("Initial grid is not 2D")
#
#    if walked_points:
#        walked_x, walked_y = zip(*walked_points)
#    else:
#        walked_x, walked_y = [], []  # Ensure empty walked points are handled
#
#    # Determine the difference (newly walked points) between initial_grid and final_custom_matrix
#    difference_coords = np.nonzero((final_custom_matrix != 0) & (initial_grid == 0))
#    difference_x, difference_y = difference_coords
#
#    # Create figure to plot both initial and new points
#    plt.figure(figsize=(8, 6))
#
#    # Plot initial points in gray
#    plt.scatter(initial_y, initial_x, c='gray', s=10, label='Initial Points')
#    
#    # Plot new points in red
#    plt.scatter(difference_y, difference_x, c='red', s=10, label='New Points')
#
#    plt.title('DLA Fractal')
#    plt.gca().invert_yaxis()  # Invert y-axis if needed
#    plt.axis('off')
#    plt.legend()
#    plt.show()
#
#    return final_custom_matrix

import numpy as np
import random
import matplotlib.pyplot as plt

def initialize_grid(custom_matrix):
    return np.where(custom_matrix > 0.01, 1, 0)

def is_isolated(grid, x, y):
    size_x, size_y = grid.shape
    neighbours = [((-1 + x) % size_x, y), ((1 + x) % size_x, y), 
                  (x, (-1 + y) % size_y), (x, (1 + y) % size_y),
                  ((-1 + x) % size_x, (-1 + y) % size_y), ((-1 + x) % size_x, (1 + y) % size_y),
                  ((1 + x) % size_x, (-1 + y) % size_y), ((1 + x) % size_x, (1 + y) % size_y)]
    
    for nx, ny in neighbours:
        if grid[nx, ny] == 0:
            return False
    return True

def random_walk(grid, initial_points, custom_matrix, max_steps, radius_range=3):
    size_x, size_y = grid.shape
    new_points = []
    walked_points = []
    
    max_custom_value = np.max(custom_matrix)
    min_custom_value = np.min(custom_matrix)

    # 根据初始值大小对初始点排序
    initial_points = sorted(initial_points, key=lambda point: custom_matrix[point[0], point[1]], reverse=True)
    
    while initial_points:
        x, y = initial_points.pop(0)  # 取出初始值最大的点
        
        if is_isolated(grid, x, y):
            continue
        
        initial_value = custom_matrix[x, y]

        # Normalize the custom_matrix value to be between 0 and 1
        normalized_value = (custom_matrix[x, y] - min_custom_value) / (max_custom_value - min_custom_value)
        # Scale the number of steps to be between 1 and max_steps, considering the column position of the point
        column_based_steps = (y / size_y) * max_steps
        steps = int(0 + normalized_value * (max_steps - 0) + column_based_steps)

        for _ in range(steps):
            if random.random() < 0.5:  # 50% 随机在周围一两格内扩展
                neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
                                (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 八个方向的邻居点
                dx, dy = random.choice(neighbours)
            else:  # 50% 随机在一定范围内探索
                angle = 2 * np.pi * random.random()
                radius = random.uniform(1, radius_range)  # 半径范围内随机选择一个点
                dx, dy = int(radius * np.cos(angle)), int(radius * np.sin(angle))

            new_x = (x + dx) % size_x
            new_y = (y + dy) % size_y
            
            walked_points.append((new_x, new_y))

            if grid[new_x, new_y] == 1:
                grid[x, y] = 1
                new_points.append((x, y))
                break

            initial_value *= 0.01  # decrease the value by a factor of 0.01 for each step
            custom_matrix[new_x, new_y] += initial_value
            x, y = new_x, new_y

    return grid, new_points, walked_points

def dla(custom_matrix, max_steps, radius_range=3):
    initial_grid = initialize_grid(custom_matrix)
    grid = initial_grid.copy()
    initial_points = list(zip(*np.nonzero(grid)))
    all_walked_points = []
    
    while initial_points:
        grid, new_points, walked_points = random_walk(grid, initial_points, custom_matrix, max_steps, radius_range)
        initial_points = new_points
        all_walked_points.extend(walked_points)
    
    return custom_matrix, initial_grid, all_walked_points

def run_dla_simulation(custom_matrix, max_steps, radius_range=5):
    final_custom_matrix, initial_grid, walked_points = dla(custom_matrix, max_steps, radius_range)

    # First plot: DLA Fractal
    plt.figure(figsize=(10, 10))
    plt.imshow(final_custom_matrix, cmap='binary')
    plt.title('DLA Fractal')
    plt.axis('off')
    plt.show()

    # Check the shape of initial_grid
    print(f"Shape of initial_grid: {initial_grid.shape}")

    initial_coords = np.nonzero(initial_grid)
    if len(initial_coords) == 2:  # Ensure it's 2D
        initial_x, initial_y = initial_coords
    else:
        raise ValueError("Initial grid is not 2D")

    if walked_points:
        walked_x, walked_y = zip(*walked_points)
    else:
        walked_x, walked_y = [], []  # Ensure empty walked points are handled

    # Determine the difference (newly walked points) between initial_grid and final_custom_matrix
    difference_coords = np.nonzero((final_custom_matrix != 0) & (initial_grid == 0))
    difference_x, difference_y = difference_coords

    # Create figure to plot both initial and new points
    plt.figure(figsize=(8, 6))

    # Plot initial points in gray
    plt.scatter(initial_y, initial_x, c='gray', s=10, label='Initial Points')
    
    # Plot new points in red
    plt.scatter(difference_y, difference_x, c='red', s=10, label='New Points')

    plt.title('DLA Fractal')
    plt.gca().invert_yaxis()  # Invert y-axis if needed
    plt.axis('off')
    plt.legend()
    plt.show()

    return final_custom_matrix
#import numpy as np
#import random
#import matplotlib.pyplot as plt
#
#def initialize_grid(custom_matrix):
#    return np.where(custom_matrix > 0.01, 1, 0)
#
#def is_isolated(grid, x, y):
#    size_x, size_y = grid.shape
#    neighbours = [((-1 + x) % size_x, y), ((1 + x) % size_x, y), 
#                  (x, (-1 + y) % size_y), (x, (1 + y) % size_y),
#                  ((-1 + x) % size_x, (-1 + y) % size_y), ((-1 + x) % size_x, (1 + y) % size_y),
#                  ((1 + x) % size_x, (-1 + y) % size_y), ((1 + x) % size_x, (1 + y) % size_y)]
#    
#    for nx, ny in neighbours:
#        if grid[nx, ny] == 0:
#            return False
#    return True
#
#def random_walk(grid, initial_points, custom_matrix, max_steps, radius_range=3):
#    size_x, size_y = grid.shape
#    new_points = []
#    walked_points = []
#    
#    max_custom_value = np.max(custom_matrix)
#    min_custom_value = np.min(custom_matrix)
#
#    # 根据初始值大小对初始点排序
#    initial_points = sorted(initial_points, key=lambda point: custom_matrix[point[0], point[1]], reverse=True)
#    
#    while initial_points:
#        x, y = initial_points.pop(0)  # 取出初始值最大的点
#        
#        if is_isolated(grid, x, y):
#            continue
#        
#        initial_value = custom_matrix[x, y]
#
#        # Normalize the custom_matrix value to be between 0 and 1
#        normalized_value = (custom_matrix[x, y] - min_custom_value) / (max_custom_value - min_custom_value)
#        # Scale the number of steps to be between 1 and max_steps, considering the column position of the point
#        column_based_steps = (y / size_y) * max_steps
#        steps = int(0 + normalized_value * (max_steps - 0) + column_based_steps)
#
#        walked_path = [(x, y)]
#        for _ in range(steps):
#            if random.random() < 0.5:  # 50% 随机在周围一两格内扩展
#                neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1),
#                                (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 八个方向的邻居点
#                dx, dy = random.choice(neighbours)
#            else:  # 50% 随机在一定范围内探索
#                angle = 2 * np.pi * random.random()
#                radius = random.uniform(1, radius_range)  # 半径范围内随机选择一个点
#                dx, dy = int(radius * np.cos(angle)), int(radius * np.sin(angle))
#
#            new_x = (x + dx) % size_x
#            new_y = (y + dy) % size_y
#            
#            walked_points.append((new_x, new_y))
#            walked_path.append((new_x, new_y))
#
#            if grid[new_x, new_y] == 1:
#                for px, py in walked_path:
#                    grid[px, py] = 1
#                    custom_matrix[px, py] = initial_value
#                new_points.append((x, y))
#                break
#
#            initial_value *= 0.01  # decrease the value by a factor of 0.01 for each step
#            custom_matrix[new_x, new_y] += initial_value
#            x, y = new_x, new_y
#
#    return grid, new_points, walked_points
#
#def add_random_points(grid, num_points):
#    size_x, size_y = grid.shape
#    available_positions = [(x, y) for x in range(size_x) for y in range(size_y) if grid[x, y] == 0]
#    
#    random_points = random.sample(available_positions, min(num_points, len(available_positions)))
#    return random_points
#
#def dla(custom_matrix, max_steps, radius_range=3, initial_random_points=100):
#    initial_grid = initialize_grid(custom_matrix)
#    grid = initial_grid.copy()
#    initial_points = list(zip(*np.nonzero(grid)))
#    random_points = add_random_points(grid, initial_random_points)
#    initial_points.extend(random_points)
#    all_walked_points = []
#    
#    while initial_points:
#        grid, new_points, walked_points = random_walk(grid, initial_points, custom_matrix, max_steps, radius_range)
#        initial_points = new_points if new_points else random_points
#        all_walked_points.extend(walked_points)
#    
#    return custom_matrix, initial_grid, all_walked_points
#
#def run_dla_simulation(custom_matrix, max_steps, radius_range=5, initial_random_points=100):
#    final_custom_matrix, initial_grid, walked_points = dla(custom_matrix, max_steps, radius_range, initial_random_points)
#
#    # First plot: DLA Fractal
#    plt.figure(figsize=(10, 10))
#    plt.imshow(final_custom_matrix, cmap='binary')
#    plt.title('DLA Fractal')
#    plt.axis('off')
#    plt.show()
#
#    # Check the shape of initial_grid
#    print(f"Shape of initial_grid: {initial_grid.shape}")
#
#    initial_coords = np.nonzero(initial_grid)
#    if len(initial_coords) == 2:  # Ensure it's 2D
#        initial_x, initial_y = initial_coords
#    else:
#        raise ValueError("Initial grid is not 2D")
#
#    if walked_points:
#        walked_x, walked_y = zip(*walked_points)
#    else:
#        walked_x, walked_y = [], []  # Ensure empty walked points are handled
#
#    # Determine the difference (newly walked points) between initial_grid and final_custom_matrix
#    difference_coords = np.nonzero((final_custom_matrix != 0) & (initial_grid == 0))
#    difference_x, difference_y = difference_coords
#
#    # Create figure to plot both initial and new points
#    plt.figure(figsize=(8, 6))
#
#    # Plot initial points in gray
#    plt.scatter(initial_y, initial_x, c='gray', s=10, label='Initial Points')
#    
#    # Plot new points in red
#    plt.scatter(difference_y, difference_x, c='red', s=10, label='New Points')
#
#    plt.title('DLA Fractal')
#    plt.gca().invert_yaxis()  # Invert y-axis if needed
#    plt.axis('off')
#    plt.legend()
#    plt.show()
#
#    return final_custom_matrix

# Example usage:
# custom_matrix = np.random.rand(100, 100)
# run_dla_simulation(custom_matrix, max_steps=500, radius_range=5, initial_random_points=200)



    



