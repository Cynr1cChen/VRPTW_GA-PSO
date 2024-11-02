import os
import math
import multiprocessing
import numpy as np
import pandas as pd
import random
from initial_pop_generation import initial_population
from initial_routing_phase2 import initial_routing_phase2
import time

def main():
    cwd = os.getcwd()
    customer_path = cwd + '\\solomon100_csv\\customers'
    dataInfo_path = cwd + '\\solomon100_csv\\data_info'
    distanceMatrix_path = cwd + '\\solomon100_csv\\distance_matrix'
    initial_path = cwd + '\\initial'

    # 目标数据集
    target_dataset = 'c101'  # 可以修改为任何需要的特定数据集
    dataset_index = 0  # 目标数据集的索引
    dataset_list = [target_dataset]

    # 文件路径列表
    customers_file = os.path.join(customer_path, f'{target_dataset}customers.csv')
    dataInfo_file = os.path.join(dataInfo_path, f'{target_dataset}dataInfo.csv')
    distanceMatrix_file = os.path.join(distanceMatrix_path, f'{target_dataset}distanceMatrix.csv')

    # 读取数据
    df_customers = pd.read_csv(customers_file)
    df_distance_matrix = pd.read_csv(distanceMatrix_file)
    df_dataInfo = pd.read_csv(dataInfo_file)

    num_customers = df_customers.shape[0]
    num_customers_depot = num_customers + 1
    df_customers.index = range(1, num_customers_depot)

    # 初始化客户数组
    arr_customers = np.empty((num_customers, 6))
    arr_customers[:, 0] = np.arange(1, num_customers_depot)
    arr_customers[:, 1] = df_customers['demand'].values
    arr_customers[:, 2] = df_customers['serviceTime'].values
    arr_customers[:, 3] = df_customers['readyTime'].values
    arr_customers[:, 4] = df_customers['dueTime'].values
    arr_customers[:, 5] = df_customers['completeTime'].values

    # 获取车辆总容量和最大工作时间
    total_capacity = df_dataInfo.loc[0, 'fleet_capacity']
    total_time = df_dataInfo.loc[0, 'fleet_max_working_time']

    # 实验变量
    num_experiments = 30
    num_cores = multiprocessing.cpu_count()  # 使用的核心数
    N = 300  # 种群大小
    greedy_percent = 0.10  # 贪心初始化的比例
    random_percent = 0.90  # 随机初始化的比例
    alpha = 100
    beta = 0.001

    all_customers_list = list(range(1, num_customers_depot))

    # 针对每个实验
    for experiment in range(num_experiments):
        radius = (np.nanmax(df_distance_matrix) - np.nanmin(df_distance_matrix)) / 2 * (random.uniform(0.5, 1))

        # 生成初始种群
        pop_chromosome_routeList_array, pop_chromosome_distanceList_array, pop_num_routes, pop_total_distance = initial_population(
            N, num_cores,
            math.floor(N * random_percent) - 1,  # 随机染色体数量
            N - (math.floor(N * random_percent) - 1) - 1,  # 贪心染色体数量
            1,  # 最近邻染色体数量
            num_customers, arr_customers, df_distance_matrix.values, total_capacity, total_time, all_customers_list,
            radius)

        # 第二阶段
        pop_chromosome_routeList_array, pop_chromosome_distanceList_array, arr_pop_results = initial_routing_phase2(
            num_cores, N, alpha, beta,
            pop_chromosome_routeList_array, pop_chromosome_distanceList_array,
            df_distance_matrix.values, arr_customers, total_time, total_capacity)

        # 保存初始解
        initial_pop_output_file = os.path.join(initial_path, f'{target_dataset}\\experiment{experiment}\\initialPop.npy')
        initial_pop_distance_output_file = os.path.join(initial_path, f'{target_dataset}\\experiment{experiment}\\initialDistance.npy')
        initial_result_output_file = os.path.join(initial_path, f'{target_dataset}\\experiment{experiment}\\initialResults.csv')

        # 创建文件夹
        initial_pop_output_dir = os.path.dirname(initial_pop_output_file)
        if not os.path.exists(initial_pop_output_dir):
            os.makedirs(initial_pop_output_dir)

        np.save(initial_pop_output_file, pop_chromosome_routeList_array)
        np.save(initial_pop_distance_output_file, pop_chromosome_distanceList_array)

        df_initial_results = pd.DataFrame(np.zeros((N, 3)), columns=['num_vehicles', 'distance', 'fitness'])
        df_initial_results['num_vehicles'] = pop_num_routes
        df_initial_results['distance'] = pop_total_distance
        df_initial_results['fitness'] = pop_num_routes + (np.arctan(pop_total_distance) / (math.pi / 2))

        df_initial_results.to_csv(initial_result_output_file, index=False)

        print('experiment: ', experiment)

    print('\n', 'Dataset:', target_dataset)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total solve time:', end_time - start_time)
