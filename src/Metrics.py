from SerialData import SerialData


class Metrics(SerialData):
    def __init__(self, accuracy: float = 0., average_train_time: float = 0., average_inference_time: float = 0., compile_time: float = 0., build_time: float = 0., save_time: float = 0.):
        self.metrics = {
            'accuracy': accuracy,
            'average_train_time': average_train_time,
            'average_inference_time': average_inference_time,
            'compile_time': compile_time,
            'build_time': build_time,
            'save_time': save_time
        }

    def serialize(self) -> dict:
        return self.metrics

    def deserialize(self, obj: dict) -> None:
        self.metrics = obj
