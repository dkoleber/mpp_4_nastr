from SerialData import SerialData


class Hyperparameters(SerialData):
    def __init__(self):
        super().__init__()

        self.parameters = Hyperparameters._default_parameters()

    def serialize(self) -> dict:
        return self.parameters

    def deserialize(self, obj: dict) -> None:
        self.parameters = Hyperparameters._default_parameters()

        if 'CELL_STACKS' not in obj:
            obj['CELL_STACKS'] = [obj['NORMAL_CELL_N'], 1]
            obj['CELL_STRIDES'] = [1, 2]
            obj['CELL_INDEX_FOR_REDUCTION'] = 1
            obj['GROUPS_PER_CELL'] = 5
            obj['OPS_PER_GROUP'] = 2
            obj['CONCATENATE_ALL'] = False
            obj['PASS_RESIDUAL'] = True


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
    def _default_parameters():
        return {
            'CELL_STACKS': [5, 1],
            'CELL_STRIDES': [1, 2],
            'CELL_INDEX_FOR_REDUCTION': 1, #on the last layer, only cells up to but not including the reduction cell are used
            'GROUPS_PER_CELL': 5,
            'OPS_PER_GROUP': 2,
            'CELL_LAYERS': 3,

            'TARGET_FILTER_DIMS': 32,
            'REDUCTION_EXPANSION_FACTOR' : 1,
            'REDUCTION_EXPANSION_BEFORE': False,
            'REDUCE_CURRENT': False,
            'PASS_RESIDUAL': True,
            'CONCATENATE_ALL': False,

            'TRAIN_EPOCHS': 1,
            'TRAIN_ITERATIONS': 16,
            'MAXIMUM_LEARNING_RATE': 0.002,
            'MINIMUM_LEARNING_RATE': 0.001,
            'USE_SGDR': True,
            'BATCH_SIZE': 16,

            'SGDR_EPOCHS_PER_RESTART': 16,
            'SGDR_LR_DECAY': .8,
            'SGDR_PERIOD_DECAY': 2,

            'DROP_PATH_CHANCE': .6,
            'DROP_PATH_TOTAL_STEPS_MULTI':2, #multiplies the supposed end of training by this factor, causing droppath to die off at a slower rate

            'IDENTITY_THRESHOLD': 0.,  # .33
        }
