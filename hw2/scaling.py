import numpy as np

def scale(arr):
    max_val = max(arr)
    min_val = min(arr)
    return [((2.0*(num - min_val))/(1.0*(max_val - min_val)))-1.0 for num in arr]

if __name__ == '__main__':
    arr = np.array([1.0,3.0,-7.0,11.0,55.0,8.0,6.0,-28.0])
    arr_scale = scale(arr)
    print arr_scale
