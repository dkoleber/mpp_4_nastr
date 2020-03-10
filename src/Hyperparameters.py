from SerialData import SerialData



class Hyperparameters(SerialData):
    def __init__(self, debug_mode: bool = False):
        super().__init__()

        self.parameters = {
            'IDENTITY_THRESHOLD': .33,
            'NORMAL_CELL_N': 5,
            'CELL_LAYERS': 3,
            'INITIAL_LAYER_DIMS': 16,
            'TRAIN_EPOCHS': 2,
            'LEARNING_RATE': 0.01,
            'STRATEGY': 'aging',
            'ROUNDS': 16,
            'POPULATION_SIZE': 16,
            'STRATEGY_SELECTION_SIZE': 8,
        }
        if debug_mode:
            self.parameters = {
                'IDENTITY_THRESHOLD': .33,
                'NORMAL_CELL_N': 1,
                'CELL_LAYERS': 2,
                'INITIAL_LAYER_DIMS': 1,
                'USE_POST_BLOCK_REDUCE': True,
                'TRAIN_EPOCHS': 1,
                'LEARNING_RATE': 0.01,
                'STRATEGY': 'aging',
                'ROUNDS': 10,
                'POPULATION_SIZE': 3,
                'STRATEGY_SELECTION_SIZE': 2,
                'LAYER_EXPANSION_FACTOR': 2,
            }

    def serialize(self) -> dict:
        return self.parameters

    def deserialize(self, obj: dict) -> None:
        self.parameters = obj

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
