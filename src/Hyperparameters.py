from SerialData import SerialData



class Hyperparameters(SerialData):
    def __init__(self, debug_mode: bool = False):
        super().__init__()

        self.parameters = Hyperparameters._default_parameters(debug_mode)

    def serialize(self) -> dict:
        return self.parameters

    def deserialize(self, obj: dict) -> None:
        self.parameters = Hyperparameters._default_parameters(False)
        for k, v in obj.items():
            if k in self.parameters:
                self.parameters[k] = v


    def __eq__(self, other):
        if isinstance(other, Hyperparameters):
            match = True
            for k, v in self.parameters.items():
                if k in other.parameters:
                    if other.parameters[k] != self.parameters[k]:
                        match = False
            return match
        else:
            return False

    @staticmethod
    def _default_parameters(debug_mode: bool):
        if not debug_mode:
            return {

                'NORMAL_CELL_N': 5,
                'CELL_LAYERS': 3,
                'INITIAL_LAYER_DIMS': 32,
                'TARGET_FILTER_DIMS': 32,

                'TRAIN_EPOCHS': 2,
                'TRAIN_ITERATIONS': 2,
                'LEARNING_RATE': 0.01,

                'SGDR_EPOCHS_PER_RESTART': 1.5,
                'SGDR_LR_DECAY': .95,
                'SGDR_PERIOD_DECAY': 1.05,

                'IDENTITY_THRESHOLD': 0.,  # .33
            }
        else:
            return {

                'NORMAL_CELL_N': 1,
                'CELL_LAYERS': 2,
                'INITIAL_LAYER_DIMS': 1,
                'TARGET_FILTER_SIZE': 32,

                'TRAIN_EPOCHS': 1,
                'TRAIN_ITERATIONS': 2,
                'LEARNING_RATE': 0.01,

                'SGDR_EPOCHS_PER_RESTART': 1.5,
                'SGDR_LR_DECAY': .95,
                'SGDR_PERIOD_DECAY': 1.05,

                'IDENTITY_THRESHOLD': .33,
            }