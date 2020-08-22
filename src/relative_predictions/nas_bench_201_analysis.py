from nas_201_api import NASBench201API as nasapi
import os
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))

from FileManagement import *

def get_api():
    file_name = 'NAS-Bench-201-v1_1-096897.pth'
    file_path = os.path.join(res_dir, file_name)

    if os.path.exists(file_path):
        api = nasapi(file_path)
        return api
    else:
        return None


def load_nasbench201_accuraices():
    accs = np.load(os.path.join(res_dir, 'nas_bench_201_cifar10_test_accuracies.npy'))
    accs = np.swapaxes(accs, 0, 1)
    return accs






if __name__ == '__main__':
    test()