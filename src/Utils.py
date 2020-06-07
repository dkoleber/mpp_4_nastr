import numpy as np

def get_random_int(max_val:int, min_val: int = 0) -> int:
    return int((np.random.random() * (max_val - min_val)) + min_val)