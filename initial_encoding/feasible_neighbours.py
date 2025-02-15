import numpy as np


def feasible_neighbours(num_customers, arr_customers, arr_distance_matrix, available_customers_list, total_time, total_capacity, curr_time, curr_capacity, curr_customer):
    arr_feasible_customers = np.empty((len(available_customers_list), 12))
    available_customers_list_index = available_customers_list.copy()
    available_customers_list_index[:] = [x - 1 for x in available_customers_list_index]
    arr_feasible_customers[:,0:6] = arr_customers[available_customers_list_index, :]
    
    #curr capacity 
    arr_feasible_customers[:, 1] = arr_feasible_customers[:, 1] + curr_capacity 
    #distance_curr_next
    arr_feasible_customers[:, 6] = arr_distance_matrix[curr_customer, available_customers_list]
    #arrival_time 
    arr_feasible_customers[:, 7] = arr_feasible_customers[:, 6] + curr_time
    #waiting_time
    arr_feasible_customers[:, 8] = arr_feasible_customers[:, 3] - arr_feasible_customers[:, 7]
    arr_feasible_customers[:, 8] = arr_feasible_customers[:, 8].clip(min=0)
    #start time
    arr_feasible_customers[:, 9] = arr_feasible_customers[:, 7] + arr_feasible_customers[:, 8] #arrival + waiting
    #finish_time 
    arr_feasible_customers[:, 10] = arr_feasible_customers[:, 9] + arr_feasible_customers[:, 2] #(arrival + waiting )+ service
    #return_time 
    arr_feasible_customers[:, 11] = arr_distance_matrix[available_customers_list, 0] + arr_feasible_customers[:, 10]
    
    
    arr_feasible_customers = arr_feasible_customers[(arr_feasible_customers[:,1]<=total_capacity) & (arr_feasible_customers[:,11]<=total_time) & (arr_feasible_customers[:,9]>=arr_feasible_customers[:,3]) & (arr_feasible_customers[:,10]<=arr_feasible_customers[:,5])]
    
    return arr_feasible_customers

    
    
    
