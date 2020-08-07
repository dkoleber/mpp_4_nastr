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

def test():

    size = 200
    key = 888

    api = get_api()

    def qu(ind):
        return api.query_by_index(ind, 'cifar10', f'{size}')[key]

    def acc(entry, ind):
        return (entry.get_eval('ori-test', ind)['accuracy'] / 100)

    accs = [[] for _ in range(size)]

    def app(i):
        query = qu(i)
        for e in range(size):
            a = acc(query, e)
            accs[e].append(a)

    for i in range(len(api)):
            app(i)

    accs = np.array(accs)
    print(accs.shape)
    np.save(os.path.join(res_dir, f'nas_bench_201_cifar10_test_accuracies_{size}.npy'), accs)

    # accs = np.swapaxes(accs, 0, 1)
    #
    # # scipy.stats.anderson(accs[-1])
    # xvals = [x for x in range(len(api))]
    #
    # plt.subplot(1, 1, 1)
    # plt.scatter(xvals, accs[-1])
    # plt.show()

    # num = len(api)
    # print(f'api size: {num}')
    # info = api.query_meta_info_by_index(1)
    # print(f'info {info}')
    # metrics = info.get_metrics('cifar10', 'test')
    # print(f'metrics: {metrics}')
    # test_acc = metrics['top1']
    # print(f'test_acc: {test_acc}')
    # # for i, arch_str in enumerate(api):
    #     print('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))




if __name__ == '__main__':
    test()