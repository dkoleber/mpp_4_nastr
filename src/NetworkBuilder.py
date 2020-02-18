import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from __future__ import annotations
from typing import List, Tuple
from enum import Enum

'''
1. Aging Selection
    1. select a set S candidates from population P
    2. select a set N candidates from S to have children (set M)
    3. mutate all candidates in M
    4. train all children in M
    5. Add all children in M to P
    6. Remove the |M| oldest candidates from P 

2. Tournament Selection
    select two individuals, kill the less fit one, and the more fit one has a child
    
3. Tournament with combination
    select 4 individuals, kill the least fit two, and the remaining two each have a child
    the children have some random replacement beteween eachother, and then are mtuated

---

initialization types:
full passthrough
random


---

search space

- number of blocks 
+ number of groups within blocks
+ number of items within groups
+ operation used by items (identity, 
+ initialization type of variables in operation (zeros, identity aka parent, random, xavier, etc) 




hyperconfigurations:
initialization scheme (full passthrough vs random)
initialization type (parent/zeros/random)


mutations:
add/remove group
add/remove item
change operation / change input 

'''

IDENTITY_THRESHOLD = .33
NORMAL_CELL_N = 3
CELL_LAYERS = 3


class Builder:
    @staticmethod
    def add_op(values):
        return None

    @staticmethod
    def get_op(op_type: OperationType, op_input):
        return 0  # TODO

    @staticmethod
    def sum(values):
        return tf.add_n(values)

    @staticmethod
    def concat(values):
        return tf.concat(values, 3)


class SerialData(ABC):
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def deserialize(self, obj:dict) -> None:
        pass


class OperationType(Enum):
    IDENTITY = 1


class OperationItem:
    def __init__(self, attachment_index: int):
        self.operation_type: OperationType = OperationType.IDENTITY
        self.attachment_index: int = attachment_index
        self.actual_attachment: int = 0

    def build_operation(self, operation_input):
        return Builder.get_op(self.operation_type, operation_input)


class Group:
    def __init__(self):
        self.operations: List[OperationItem] = []

    def build_group(self, available_inputs):
        outputs = []
        attachments = []
        for operation in self.operations:
            built_op = operation.build_operation(available_inputs[operation.actual_attachment])
            outputs.append(built_op)
            attachments.append(operation.actual_attachment)
        return Builder.sum(outputs), attachments


class Block:
    def __init__(self, num_inputs: int):
        self.groups: List[Group] = []
        self.num_inputs = num_inputs
    # def mutate(self):
    # choice = np.random.random()
        # thresholds = [0,0]
        # if choice < thresholds[0]:
        #     pass  # identity
        # elif thresholds[0] < choice < thresholds[1]:
        #     pass  # add/remove block TODO
        # elif thresholds[1] < choice < thresholds[2]:
        #
        #     pass  # alter block

    def build_block(self, block_inputs):
        available_inputs = block_inputs
        attachments = [False for x in range(len(self.groups) + len(block_inputs))]
        for group in self.groups:
            group_output, group_attachments = group.build_group(available_inputs)
            available_inputs.append(group_output)
            for attachment in group_attachments:
                attachments[attachment] = True
            # attachments[group.index + len(block_inputs)] = False  # this is implicit since it starts as false
        unattached = [available_inputs[x] for x in range(len(attachments)) if not attachments[x]]
        return Builder.concat(unattached)


class Model:
    def __init__(self, ):
        self.blocks: List[Block] = []  # [NORMAL_CELL, REDUCTION_CELL]

    def populate_with_NASnet_blocks(self):
        def get_block():
            block = Block(2)
            block.groups = [Group() for x in range(5)]
            for i in range(5):
                block.groups[i].operations = [OperationItem(i + 2) for x in range(2)]  # +2 because 2 inputs for block, range(2) because pairwise groups
                # TODO set operations
            return block
        self.blocks.append(get_block())  # normal block
        self.blocks.append(get_block())  # reduction block

    def build_graph(self, graph_input):
        block_ops = []
        for block in self.blocks:
            block_ops.append(block.build_block)

        block_input = graph_input
        for layer in range(CELL_LAYERS):
            for normal_cells in range(NORMAL_CELL_N):
                block_input = block_ops[0](block_input)
            block_input = block_ops[1](block_input)

        return block_input

    def mutate(self):

        other_mutation_threshold = ((1 - IDENTITY_THRESHOLD) / 2.) + IDENTITY_THRESHOLD

        select_block = self.blocks[int(np.random.random() * len(self.blocks))]
        select_group = select_block.groups[int(np.random.random() * len(select_block.groups))]
        select_item = select_group.operations[int(np.random.random() * len(select_group.operations))]
        select_mutation = np.random.random()
        if select_mutation < IDENTITY_THRESHOLD:
            return
        elif IDENTITY_THRESHOLD < select_mutation < other_mutation_threshold:
            # hidden state mutation = change inputs
            select_item.actual_attachment = int(np.random.random() * select_item.attachment_index)
        else:
            select_item.operation_type = int(np.random.random() * OperationType.IDENTITY)


class Candidate:
    def __init__(self):
        self.fitness = 0
        self.age = 0

    def evaluate_fitness(self) -> None:
        pass

    def duplicate(self) -> Candidate:
        pass

    def mutate(self) -> None:
        pass
        # for block in NUM_BLOCKS:
        #     for


class EvolutionStrategy(ABC):
    @abstractmethod
    def evolve_population(self, population: List[Candidate]) -> Tuple[List[Candidate], List[Candidate], List[Candidate]]:
        """
        Evolves a population
        :param population: The population to evolve
        :return: A tuple containing candidates that (carried over but weren't changed, were added, were removed)
        """
        pass


class AgingStrategy(EvolutionStrategy):
    def __init__(self, sample_size: int = 2):
        super().__init__()
        self.sample_size = sample_size

    def evolve_population(self, population: List[Candidate]):
        sampled_candidates = [population[x] for x in np.random.randint(0, len(population), size=self.sample_size)]
        sampled_fitness = [x.fitness for x in sampled_candidates]
        best_candidate = int(np.argmax(sampled_fitness))
        new_candidate = sampled_candidates[best_candidate].duplicate()
        new_candidate.mutate()
        population.append(new_candidate)
        return population[1:]


def main():
    rounds = 10
    population_size = 10

    evolution_strategy = AgingStrategy()
    population = [Candidate() for _ in range(population_size)]
    history = population
    for r in range(rounds):
        population, new_candidates, removed_candidates = evolution_strategy.evolve_population(population)
        history.extend(new_candidates)

    history_fitness = [x.fitness for x in history]
    best_candidate = int(np.argmax(history_fitness))
    return history[best_candidate]
