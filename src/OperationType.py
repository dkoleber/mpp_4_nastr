from enum import IntEnum


class OperationType(IntEnum):
    IDENTITY = 0,
    SEP_3X3 = 1,
    SEP_5X5 = 2,
    SEP_7X7 = 3,
    AVG_3X3 = 4,
    MAX_3X3 = 5,
    DIL_3X3 = 6,
    SEP_1X7_7X1 = 7

    @staticmethod
    def lookup_string(val):
        switch = {
            OperationType.IDENTITY: 'IDENTITY',
            OperationType.SEP_3X3: 'SEP_3X3',
            OperationType.SEP_5X5: 'SEP_5X5',
            OperationType.SEP_7X7: 'SEP_7X7',
            OperationType.AVG_3X3: 'AVG_3X3',
            OperationType.MAX_3X3: 'MAX_3X3',
            OperationType.DIL_3X3: 'DIL_3X3',
            OperationType.SEP_1X7_7X1: 'SEP_1X7_7X1'
        }
        return switch[val]
