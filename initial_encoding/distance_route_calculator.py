def distance_calculator(route, arr_disatnce_matrix):
    distance = 0
    for i in range(len(route)-1):
        distance+=arr_disatnce_matrix[route[i], route[i+1]]
        
    return distance
