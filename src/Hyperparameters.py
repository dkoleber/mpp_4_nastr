from SerialData import SerialData


class Hyperparameters(SerialData):
    def __init__(self):
        super().__init__()
        self.parameters = {
            'IDENTITY_THRESHOLD': .33,
            'NORMAL_CELL_N': 5,
            'CELL_LAYERS': 3,
            'INITIAL_LAYER_DIMS': 16,
            'USE_POST_BLOCK_REDUCE': True,
            'TRAIN_EPOCHS': 2,
            'LEARNING_RATE': 0.001,
            'STRATEGY': 'aging',
            'ROUNDS': 32,
            'POPULATION_SIZE': 32,
			'STRATEGY_SELECTION_SIZE': 8,
        }
        #self.parameters = {
        #    'IDENTITY_THRESHOLD': .33,
        #    'NORMAL_CELL_N': 1,
        #    'CELL_LAYERS': 1,
        #    'INITIAL_LAYER_DIMS': 1,
        #    'USE_POST_BLOCK_REDUCE': True,
        #    'TRAIN_EPOCHS': 1,
        #    'LEARNING_RATE': 0.001,
        #    'STRATEGY': 'aging',
        #    'ROUNDS': 1,
        #    'POPULATION_SIZE': 2,
        #    'STRATEGY_SELECTION_SIZE': 2,
        #}

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
