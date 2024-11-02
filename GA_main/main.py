import os
import numpy as np
import pandas as pd
import multiprocessing
import time
import math
import copy
from functions import full_recombination, mutation, elitism
import matplotlib.pyplot as plt

start_time_total = time.time()
# Paths
cwd = os.getcwd()
initial_path = os.path.abspath(os.path.join(cwd, '..', 'initial_encoding', 'initial'))
final_path = os.path.join(cwd, 'final')

# 目标数据集
target_dataset = 'c101'

# 文件路径
customer_path = os.path.join(cwd, 'solomon100_csv', 'customers')
dataInfo_path = os.path.join(cwd, 'solomon100_csv', 'data_info')
distanceMatrix_path = os.path.join(cwd, 'solomon100_csv', 'distance_matrix')

# 读取数据
customer_file = os.path.join(customer_path, f'{target_dataset}customers.csv')
dataInfo_file = os.path.join(dataInfo_path, f'{target_dataset}dataInfo.csv')
distanceMatrix_file = os.path.join(distanceMatrix_path, f'{target_dataset}distanceMatrix.csv')

df_customers = pd.read_csv(customer_file)
df_distance_matrix = pd.read_csv(distanceMatrix_file)
df_dataInfo = pd.read_csv(dataInfo_file)

num_customers = df_customers.shape[0]
num_customers_depot = num_customers + 1
df_customers.index = range(1, num_customers_depot)

arr_distance_matrix = np.asarray(df_distance_matrix)
dataset_radius = (np.nanmax(arr_distance_matrix) - np.nanmin(arr_distance_matrix)) / 2

arr_customer_attributes = ['demand', 'readyTime', 'dueTime']
arr_customers = np.empty((num_customers, len(arr_customer_attributes) + 1), dtype=int)
arr_customers[:, 0] = np.arange(1, num_customers_depot, 1, dtype=int)
arr_customers[:, 1:] = df_customers[arr_customer_attributes]

total_time = df_dataInfo.loc[0, 'fleet_max_working_time']
total_capacity = df_dataInfo.loc[0, 'fleet_capacity']
service_time = df_customers.loc[1, 'serviceTime']

# 实验变量
num_experiments = 3
num_cores = multiprocessing.cpu_count()
N = 300
num_generations = 350
num_elite_chromosomes = 1
K_val = 4
r = 0.8

# 初始化实验结果路径
experiment_results_file = os.path.join(final_path, f'{target_dataset}', f'{target_dataset}_experiment_results.csv')
experiment_route_file = os.path.join(final_path, f'{target_dataset}', f'{target_dataset}_experiment_routes.npy')
experiment_distance_file = os.path.join(final_path, f'{target_dataset}', f'{target_dataset}_experiment_distances.npy')

# 创建输出目录
os.makedirs(os.path.dirname(experiment_results_file), exist_ok=True)
os.makedirs(os.path.dirname(experiment_route_file), exist_ok=True)
os.makedirs(os.path.dirname(experiment_distance_file), exist_ok=True)

experimental_routes = np.empty(num_experiments, object)
experimental_distances = np.empty(num_experiments, object)
experimental_fitness_values = []  # 用于存储每个实验的适应度值列表

df_global_result = pd.DataFrame(np.zeros((1, 4)), columns=['num_vehicles', 'distance', 'fitness', 'time'])
df_experimental_results = pd.DataFrame(np.zeros((num_experiments, 4)), columns=['num_vehicles', 'distance', 'fitness', 'time'])

for experiment in range(num_experiments):
    print(f'Experiment {experiment} started...')
    initial_pop_output_file = os.path.join(initial_path, f'{target_dataset}', f'experiment{experiment}', 'initialPop.npy')
    initial_pop_distance_output_file = os.path.join(initial_path, f'{target_dataset}', f'experiment{experiment}', 'initialDistance.npy')
    initial_result_output_file = os.path.join(initial_path, f'{target_dataset}', f'experiment{experiment}', 'initialResults.csv')

    pop_chromosome_routeList_array = np.load(initial_pop_output_file, allow_pickle=True)
    pop_chromosome_distanceList_array = np.load(initial_pop_distance_output_file, allow_pickle=True)
    df_pop_results = pd.read_csv(initial_result_output_file)
    arr_pop_results = np.asarray(df_pop_results)

    # 在每个实验内部初始化适应度值列表
    fitness_values = []

    start_time = time.time()
    for gen in range(num_generations):
        new_pop_chromosome_routeList_array = np.empty(N, object)
        new_pop_chromosome_distanceList_array = np.empty(N, object)
        arr_new_pop_results = np.empty((N, 3), float)

        arr_new_pop_results, new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array = full_recombination(
            num_cores, K_val, r, N, arr_pop_results, arr_new_pop_results, arr_customers, arr_distance_matrix,
            total_time, total_capacity, pop_chromosome_routeList_array, pop_chromosome_distanceList_array,
            new_pop_chromosome_routeList_array, new_pop_chromosome_distanceList_array, service_time
        )

        elite_chromosome_routeList, elite_chromosome_distanceList, elite_result = elitism(
            pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results
        )

        worst_chromosome_number = np.nanargmax(arr_new_pop_results[:, 2])
        new_pop_chromosome_routeList_array[worst_chromosome_number] = elite_chromosome_routeList
        new_pop_chromosome_distanceList_array[worst_chromosome_number] = elite_chromosome_distanceList
        arr_new_pop_results[worst_chromosome_number, :] = elite_result[:]

        pop_chromosome_routeList_array = copy.deepcopy(new_pop_chromosome_routeList_array)
        pop_chromosome_distanceList_array = copy.deepcopy(new_pop_chromosome_distanceList_array)
        arr_pop_results = copy.deepcopy(arr_new_pop_results)

        # 记录当前代数的最优适应度值
        min_fitness = np.nanmin(arr_pop_results[:, 2])
        fitness_values.append(min_fitness)

    end_time = time.time()
    print('Time:', end_time - start_time, 's')

    df_experimental_results.iloc[experiment, 0:3] = arr_pop_results[np.nanargmin(arr_pop_results[:, 2]), :]
    df_experimental_results.loc[experiment, 'time'] = end_time - start_time

    experimental_routes[experiment] = copy.deepcopy(pop_chromosome_routeList_array[np.nanargmin(arr_pop_results[:, 2])])
    experimental_distances[experiment] = copy.deepcopy(pop_chromosome_distanceList_array[np.nanargmin(arr_pop_results[:, 2])])

    # 将当前实验的适应度值列表添加到总列表中
    experimental_fitness_values.append(fitness_values)

    print('Experiment:', experiment, 'finished', '\n')

# 在所有实验结束后，提取全局最佳结果
df_global_result.iloc[0, :] = df_experimental_results.iloc[df_experimental_results['fitness'].idxmin(), :]

fitness_index = df_experimental_results['fitness'].idxmin()
gbest_route = experimental_routes[fitness_index]
best_fitness_values = experimental_fitness_values[fitness_index]

print(df_experimental_results)
print('Dataset:', target_dataset)

end_time_total = time.time()
print('Total solve time:', end_time_total - start_time_total, 's')

# 仓库坐标
depot_x = df_dataInfo.loc[0, 'fleet_start_x_coord']
depot_y = df_dataInfo.loc[0, 'fleet_start_y_coord']

# 绘制最优实验的路径图和适应度值迭代图
def plot_route_and_fitness(customers, route, depot_x, depot_y, fitness_values):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 创建 1x2 的子图布局

    # 获取客户坐标，设置索引为 'Customer' 列
    customer_coords = customers.set_index('Customer')[['XC', 'YC']]

    # 绘制客户位置
    axs[0].scatter(customer_coords['XC'], customer_coords['YC'], c='blue', label='Customers', s=50, edgecolor='black', zorder=4)
    # 绘制仓库位置
    axs[0].scatter(depot_x, depot_y, c='red', label='Depot', marker='D', s=100, edgecolor='black', zorder=5)

    # 显示客户编号
    for customer_id, coords in customer_coords.iterrows():
        axs[0].text(coords['XC'], coords['YC'], str(int(customer_id)), fontsize=8, ha='right', zorder=6)

    # 为每条路线分配不同的颜色
    colors = plt.cm.get_cmap('tab20', len(route))

    # 绘制每条路线
    for idx, path in enumerate(route):
        route_x = []
        route_y = []
        for customer in path:
            if customer == 0:
                x = depot_x
                y = depot_y
            else:
                x = customer_coords.loc[customer, 'XC']
                y = customer_coords.loc[customer, 'YC']
            route_x.append(x)
            route_y.append(y)
        axs[0].plot(route_x, route_y, marker='o', color=colors(idx), label=f'Route {idx+1}')

    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].set_title('Optimal VRPTW Route')
    axs[0].legend()
    axs[0].grid(True)

    # 适应度值迭代图
    axs[1].plot(range(1, len(fitness_values) + 1), fitness_values, color='purple', marker='o', linestyle='-', linewidth=1.5, markersize=4)
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Fitness Value')
    axs[1].set_title('Fitness Value Over Generations')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# 使用最优结果数据绘图
plot_route_and_fitness(df_customers, gbest_route, depot_x, depot_y, best_fitness_values)
