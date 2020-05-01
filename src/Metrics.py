from typing import List

from SerialData import SerialData


class Metrics(SerialData):
    def __init__(self, accuracy: List[float] = None, average_train_time: List[float] = None, average_inference_time: List[float] = None, compile_time: float = 0., build_time: float = 0., save_time: float = 0.):

        self.metrics = {
            'accuracy': accuracy if accuracy is not None else [],
            'average_train_time': average_train_time if average_train_time is not None else [],
            'average_inference_time': average_inference_time if average_inference_time is not None else [],
            'compile_time': compile_time,
            'build_time': build_time,
            'save_time': save_time,
            'mutation_info':
                {
                    'parent':'',
                    'cell_index':0,
                    'group_index':0,
                    'item_index':0,
                    'mutation_type':0,
                    'mutation_subtype':0
                },
            'flops':0
        }


    def serialize(self) -> dict:
        return self.metrics

    def deserialize(self, obj: dict) -> None:
        for k, v in obj.items():
            if k in self.metrics:
                self.metrics[k] = v
