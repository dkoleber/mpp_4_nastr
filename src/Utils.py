import numpy as np

def get_random_int(max_val:int, min_val: int = 0) -> int:
    return int((np.random.random() * (max_val - min_val)) + min_val)


def list_contains_list(list_of_lists, list_to_check):
    contains = False
    for sub_list in list_of_lists:
        if len(sub_list) != len(list_to_check):
            continue
        mismatch = False
        for index in range(len(sub_list)):
            if sub_list[index] != list_to_check[index]:
                mismatch = True
                break

        if not mismatch:
            return True
    return False

def get_multi_dim(dims):
    if len(dims) == 1:
        return [None for _ in range(dims[0])]
    else:
        return [get_multi_dim(dims[1:]) for _ in range(dims[0])]