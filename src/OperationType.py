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

