import numpy as np

def compute_moving_avg(data_list):
    N = 10
    mov_avg = np.convolve(np.array(data_list), np.ones(N) / N, mode='valid')
    return mov_avg