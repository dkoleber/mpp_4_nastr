from __future__ import annotations
from enum import IntEnum


class OperationType(IntEnum):
    IDENTITY = 0,
    SEP_3X3 = 1,
    SEP_5X5 = 2,
    SEP_7X7 = 3,
    AVG_3X3 = 4,
    AVG_5X5 = 5,
    MAX_3X3 = 6,
    MAX_5X5 = 7
    CONV_3X3 = 8,
    SEP_1X7_7X1 = 9

    @staticmethod
    def lookup_string(val):
        switch = {
            OperationType.IDENTITY: 'IDENTITY',
            OperationType.SEP_3X3: 'SEP_3X3',
            OperationType.SEP_5X5: 'SEP_5X5',
            OperationType.SEP_7X7: 'SEP_7X7',
            OperationType.AVG_3X3: 'AVG_3X3',
            OperationType.AVG_5X5: 'AVG_5X5',
            OperationType.MAX_3X3: 'MAX_3X3',
            OperationType.MAX_5X5: 'MAX_5X5',
            OperationType.CONV_3X3: 'CONV_3X3',
            OperationType.SEP_1X7_7X1: 'SEP_1X7_7X1'
        }
        return switch[val]

    @staticmethod
    def lookup_opname(op:OperationType, requires_reduction: bool = False):
        if op == OperationType.IDENTITY:
            if requires_reduction:
                return 'identity_reduction_operation'
            else:
                return 'identity_operation'

        vals = {
            OperationType.SEP_3X3: 'seperable_convolution_operation',
            OperationType.SEP_5X5: 'seperable_convolution_operation',
            OperationType.SEP_7X7: 'seperable_convolution_operation',
            OperationType.AVG_3X3: 'average_pooling_operation',
            OperationType.AVG_5X5: 'average_pooling_operation',
            OperationType.MAX_3X3: 'max_pooling_operation',
            OperationType.MAX_5X5: 'max_pooling_operation',
            OperationType.CONV_3X3: 'convolutional_operation',
            OperationType.SEP_1X7_7X1: 'doubly_seperable_convolution_operation'
        }
        return vals[op]