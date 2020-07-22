from nas_201_api import NASBench201API as nasapi
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,'..\\'))

from FileManagement import *

def test():
    file_name = 'NAS-Bench-201-v1_1-096897.pth'
    file_path = os.path.join(res_dir, file_name)

    if os.path.exists(file_path):
        api = nasapi(file_path)

        num = len(api)
        print(f'api size: {num}')
        info = api.query_meta_info_by_index(1)
        print(f'info {info}')
        metrics = info.get_metrics('cifar10', 'test')
        print(f'metrics: {metrics}')
        test_acc = metrics['top1']
        print(f'test_acc: {test_acc}')
        # for i, arch_str in enumerate(api):
        #     print('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))


if __name__ == '__main__':
    test()