import numpy as np
import pandas as pd
import copy
import os
import time
import multiprocessing
import math
import matplotlib.pyplot as plt
from functions import (
    omega,
    pbest_particle_pC_list,
    PSO,
    CLPSO_velocity_update,
    update_route_position,
    PSO_result_updater,
    CLPSO_result_updater,
    local_search_result_updater,
    global_result_from_experiments,
    local_search
)

start_time_total = time.time()
# 参数设置
num_customers = 100
num_experiments = 3  # 根据需要设置实验次数

# 路径设置
cwd = os.getcwd()
# print(cwd)
initial_path = os.path.abspath(os.path.join(cwd, '..', 'initial_encoding', 'initial'))
# print(f'Initial path: {initial_path}')
final_path = os.path.join(cwd, 'final')
# print(f'Final path: {final_path}')

customer_path = os.path.join(cwd, 'solomon100_csv', 'customers')
dataInfo_path = os.path.join(cwd, 'solomon100_csv', 'data_info')
distanceMatrix_path = os.path.join(cwd, 'solomon100_csv', 'distance_matrix')

# 目标数据集
target_dataset = 'c101'

# 文件路径
customer_file = os.path.join(customer_path, f'{target_dataset}customers.csv')
dataInfo_file = os.path.join(dataInfo_path, f'{target_dataset}dataInfo.csv')
distanceMatrix_file = os.path.join(distanceMatrix_path, f'{target_dataset}distanceMatrix.csv')

# 读取数据
df_customers = pd.read_csv(customer_file)
df_distance_matrix = pd.read_csv(distanceMatrix_file)
arr_distance_matrix = df_distance_matrix.values
df_data_information = pd.read_csv(dataInfo_file)
num_cust = df_customers.shape[0]
num_customers_depot = num_cust + 1  # 包括仓库节点
df_customers.index = range(1, num_cust + 1)  # 重置索引

# 初始化客户数组
arr_customer_cols = 15
arr_customers = np.empty([num_cust, arr_customer_cols])
arr_customers[:, 0] = df_customers.index  # 客户编号
arr_customers[:, [1, 2, 3, 4, 13]] = df_customers.loc[:, ['demand', 'readyTime', 'dueTime', 'serviceTime', 'completeTime']].values

arr_customer_info = arr_customers[:, [0, 1, 2, 3]]  # 0:客户编号, 1:需求, 2:最早时间, 3:最晚时间

# 全局变量
total_capacity = df_data_information.loc[0, 'fleet_capacity']
total_time = df_data_information.loc[0, 'fleet_max_working_time']
service_time = df_customers.loc[1, 'serviceTime']

# 实验变量
num_cores = multiprocessing.cpu_count()
M = 300  # 种群大小，与GA初始化代码中的N一致
sg = 1000  # 停止条件
rg = 7  # 刷新间隔
c = 2  # 常数c
phi = 0.3
w0 = 0.9
w1 = 0.4

max_gen = sg
denom = (math.exp(10) - 1)
M_1 = M - 1

# 初始化结果数据框
df_global_result = pd.DataFrame(columns=['num_vehicles', 'distance', 'fitness', 'time'])
df_experimental_results = pd.DataFrame(columns=['num_vehicles', 'distance', 'fitness', 'time'])

# 定义变量列索引
distance_col = 1
num_vehicles_col = 0
fitness_col = 2

# 开始实验
dict_routes = {}
experimental_fitness_values = []  # 用于存储每个实验的适应度值列表

for experiment in range(num_experiments):
    print(f'Experiment {experiment} started...')
    # 读取GA初始化生成的初始解
    initial_pop_output_file = os.path.join(initial_path, f'{target_dataset}', f'experiment{experiment}', 'initialPop.npy')
    initial_pop_distance_output_file = os.path.join(initial_path, f'{target_dataset}', f'experiment{experiment}', 'initialDistance.npy')
    initial_result_output_file = os.path.join(initial_path, f'{target_dataset}', f'experiment{experiment}', 'initialResults.csv')

    # 检查文件是否存在
    if not os.path.exists(initial_pop_output_file):
        print(f"Initial population file not found: {initial_pop_output_file}")
        continue

    # 加载初始种群
    pop_particle_routeList_list = np.load(initial_pop_output_file, allow_pickle=True)
    pop_particle_distance_list = np.load(initial_pop_distance_output_file, allow_pickle=True)
    df_results = pd.read_csv(initial_result_output_file)
    arr_results = df_results[['num_vehicles', 'distance', 'fitness']].values  # ['num_vehicles', 'distance', 'fitness']
    pop_distance_list = arr_results[:, distance_col]
    pop_num_route_list = arr_results[:, num_vehicles_col]

    # 初始化粒子的位置和速度
    # 将路线列表转换为邻接矩阵表示
    pop_particle_position_list = []
    num_customers_actual = num_customers_depot - 1  # 实际客户数量，不包括仓库节点

    for chromosome in pop_particle_routeList_list:
        position_matrix = np.zeros((num_customers_depot, num_customers_depot))
        for route in chromosome:
            for i in range(len(route) - 1):
                from_node = int(route[i])
                to_node = int(route[i + 1])
                position_matrix[from_node, to_node] = 1
        pop_particle_position_list.append(position_matrix)

    pop_particle_position_list = np.array(pop_particle_position_list)
    pop_particle_velocity_list = np.zeros_like(pop_particle_position_list)

    # 初始化pbest
    pbest_particle_routeList_list = copy.deepcopy(pop_particle_routeList_list)
    pbest_particle_distance_list = copy.deepcopy(pop_particle_distance_list)
    pbest_particle_position_list = copy.deepcopy(pop_particle_position_list)
    pbest_particle_velocity_list = copy.deepcopy(pop_particle_velocity_list)
    arr_pbest_results = copy.deepcopy(arr_results)
    pbest_distance_list = pop_distance_list.copy()
    pbest_num_route_list = pop_num_route_list.copy()

    # 初始化gbest
    gbest_particle_no = np.argmin(arr_pbest_results[:, fitness_col])  # 最小的fitness值对应的粒子编号
    gbest_routeList_list = pop_particle_routeList_list[gbest_particle_no]
    gbest_distance_list = pop_particle_distance_list[gbest_particle_no]
    gbest_position_list = pbest_particle_position_list[gbest_particle_no]
    gbest_velocity_list = pbest_particle_velocity_list[gbest_particle_no]
    arr_gbest_result = arr_pbest_results[gbest_particle_no, :]
    gbest_fitness = arr_gbest_result[fitness_col]
    gbest_NV = arr_gbest_result[num_vehicles_col]
    gbest_dist = arr_gbest_result[distance_col]

    # 初始化其他变量
    flag = np.zeros(M)
    gbest_tracker = 0
    iteration_no = 0

    # 初始化适应度值列表
    fitness_over_iterations = []  # 用于存储每次迭代的 fitness 值

    # 开始PSO算法迭代
    start_timer = time.time()
    while gbest_tracker < sg:
        w = omega(w0, w1, gbest_tracker, max_gen)
        arr_pbest_particle = pbest_particle_pC_list(M, denom, M_1)

        # PSO部分
        if np.max(flag) >= rg:
            PSO_resultlist, flagged_indices = PSO(
                flag, rg, w, c, num_cores, num_customers_depot, arr_distance_matrix,
                arr_customers, total_time, total_capacity, gbest_position_list,
                pbest_particle_position_list, pop_particle_position_list, pop_particle_velocity_list
            )

            # 更新结果
            (
                flag, arr_results, arr_pbest_results, pop_particle_routeList_list,
                pop_particle_position_list, pop_particle_distance_list, pop_distance_list,
                pop_num_route_list, pop_particle_velocity_list, arr_gbest_result,
                gbest_velocity_list, gbest_position_list, gbest_routeList_list,
                gbest_distance_list, gbest_tracker, pbest_particle_routeList_list,
                pbest_particle_position_list, pbest_particle_distance_list, pbest_distance_list,
                pbest_num_route_list, pbest_particle_velocity_list
            ) = PSO_result_updater(
                rg, PSO_resultlist, arr_results, arr_pbest_results, flag, flagged_indices,
                pop_particle_routeList_list, pop_particle_position_list, pop_particle_distance_list,
                pop_distance_list, pop_num_route_list, pop_particle_velocity_list, arr_gbest_result,
                gbest_velocity_list, gbest_position_list, gbest_routeList_list, gbest_distance_list,
                gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,
                pbest_particle_distance_list, pbest_distance_list, pbest_num_route_list,
                pbest_particle_velocity_list
            )

        # CLPSO部分
        # 更新速度
        pop_particle_velocity_list = CLPSO_velocity_update(
            M, w, c, num_customers_depot, pop_particle_position_list,
            pbest_particle_position_list, pop_particle_velocity_list
        )

        # 更新位置
        position_resultslist = update_route_position(
            num_cores, arr_customers, arr_distance_matrix, total_time, total_capacity,
            num_customers_depot, pop_particle_velocity_list, pop_particle_position_list, M
        )

        # 更新结果
        (
            flag, arr_results, arr_pbest_results, pop_particle_routeList_list,
            pop_particle_position_list, pop_particle_distance_list, pop_distance_list,
            pop_num_route_list, arr_gbest_result, gbest_velocity_list, gbest_position_list,
            gbest_routeList_list, gbest_distance_list, gbest_tracker, pbest_particle_routeList_list,
            pbest_particle_position_list, pbest_particle_distance_list, pbest_distance_list,
            pbest_num_route_list, pbest_particle_velocity_list
        ) = CLPSO_result_updater(
            M, arr_results, arr_pbest_results, flag, pop_particle_velocity_list,
            position_resultslist, arr_gbest_result, gbest_velocity_list, gbest_position_list,
            gbest_routeList_list, gbest_distance_list, gbest_tracker, pbest_particle_routeList_list,
            pbest_particle_position_list, pbest_particle_distance_list, pbest_distance_list,
            pbest_num_route_list, pbest_particle_velocity_list
        )

        # 本地搜索（可选）
        local_search_results_list = local_search(
            M, num_cores, df_customers, arr_distance_matrix, total_time, total_capacity,
            pop_distance_list, pop_num_route_list, pop_particle_routeList_list,
            pop_particle_distance_list, pop_particle_position_list, pop_particle_velocity_list,
            arr_customer_info, service_time
        )

        # 更新结果
        (
            flag, arr_results, arr_pbest_results, pop_particle_routeList_list,
            pop_particle_position_list, pop_particle_velocity_list, pop_particle_distance_list,
            pop_distance_list, pop_num_route_list, arr_gbest_result, gbest_velocity_list,
            gbest_position_list, gbest_routeList_list, gbest_distance_list, gbest_tracker,
            pbest_particle_routeList_list, pbest_particle_position_list,
            pbest_particle_distance_list, pbest_distance_list, pbest_num_route_list,
            pbest_particle_velocity_list
        ) = local_search_result_updater(
            local_search_results_list, flag, arr_results, arr_pbest_results,
            pop_particle_velocity_list, pop_particle_routeList_list, pop_particle_position_list,
            pop_particle_distance_list, pop_distance_list, pop_num_route_list, arr_gbest_result,
            gbest_velocity_list, gbest_position_list, gbest_routeList_list, gbest_distance_list,
            gbest_tracker, pbest_particle_routeList_list, pbest_particle_position_list,
            pbest_particle_distance_list, pbest_distance_list, pbest_num_route_list,
            pbest_particle_velocity_list
        )

        # 记录当前迭代的最佳 fitness 值
        fitness_over_iterations.append(arr_gbest_result[fitness_col])

        iteration_no += 1
        gbest_tracker += 1

    end_timer = time.time()

    # 记录实验结果
    df_experimental_results.loc[len(df_experimental_results)] = {
        'num_vehicles': arr_gbest_result[num_vehicles_col],
        'distance': arr_gbest_result[distance_col],
        'fitness': arr_gbest_result[fitness_col],
        'time': end_timer - start_timer
    }

    dict_routes[experiment] = gbest_routeList_list
    experimental_fitness_values.append(fitness_over_iterations)

    print('time:', end_timer - start_timer, 's')
    print('experiment:', experiment, 'finished', '\n')

# 在所有实验结束后，汇总结果
# 获取最优实验的索引
best_experiment_index = df_experimental_results['fitness'].idxmin()
best_route_list = dict_routes[best_experiment_index]
best_fitness_values = experimental_fitness_values[int(best_experiment_index)]

# 输出实验结果
print(df_experimental_results)
print('Dataset:', target_dataset)

# 保存最终结果
final_result_file = os.path.join(final_path, f'{target_dataset}', 'final_result.csv')
final_route_file = os.path.join(final_path, f'{target_dataset}', 'final_route.npy')

df_experimental_results.to_csv(final_result_file, index=False)
np.save(final_route_file, dict_routes)

end_time_total = time.time()
print("Total solve time:", end_time_total - start_time_total, "s")

# 定义绘图函数
def plot_route_and_fitness(customers, route, depot_x, depot_y, fitness_values):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1：绘制路径图
    ax1 = axes[0]

    # 获取客户坐标
    customer_coords = customers.set_index('Customer')[['XC', 'YC']]

    # 绘制客户位置
    ax1.scatter(customer_coords['XC'], customer_coords['YC'], c='blue', label='Customers', s=50, edgecolor='black', zorder=4)
    # 绘制仓库位置
    ax1.scatter(depot_x, depot_y, c='red', label='Depot', marker='D', s=100, edgecolor='black', zorder=5)

    # 显示客户编号
    for customer_id, coords in customer_coords.iterrows():
        ax1.text(coords['XC'], coords['YC'], str(int(customer_id)), fontsize=8, ha='right', zorder=6)

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
        ax1.plot(route_x, route_y, marker='o', color=colors(idx), label=f'Route {idx+1}')

    ax1.set_title('Final Routes')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)

    # 子图2：绘制最优实验的 fitness 变化图
    ax2 = axes[1]
    ax2.plot(range(len(fitness_values)), fitness_values, color='green', marker='o', linestyle='-', linewidth=1.5, markersize=4)
    ax2.set_title('Fitness Evolution')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness Value')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# 调用绘图函数
# 获取仓库（配送中心）坐标
depot_x = df_data_information.loc[0, 'fleet_start_x_coord']
depot_y = df_data_information.loc[0, 'fleet_start_y_coord']

plot_route_and_fitness(df_customers, best_route_list, depot_x, depot_y, best_fitness_values)
