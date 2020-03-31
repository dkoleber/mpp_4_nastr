from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from Dataset import Dataset
from FitnessCalculator import AccuracyCalculator
from Modelv3 import MetaModel


class EvolutionStrategy(ABC):
    @abstractmethod
    def evolve_population(self, population: List[MetaModel]) -> Tuple[List[MetaModel], List[MetaModel], List[MetaModel]]:
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

    def evolve_population(self, population: List[MetaModel]):
        sampled_candidates = [population[x] for x in np.random.randint(0, len(population), size=self.sample_size)]  # TODO: non-overlapping
        sampled_fitness = [x.fitness for x in sampled_candidates]
        best_candidate_index = int(np.argmax(sampled_fitness))
        best_candidate = sampled_candidates[best_candidate_index]
        print(f'producing child of model {best_candidate.model_name}')
        new_candidate = best_candidate.produce_child()
        new_candidate.mutate()
        population.append(new_candidate)
        return population[1:], [new_candidate], [population[0]]


class SeepingStrategy(EvolutionStrategy):
    def __init__(self, seep_iterations):
        super().__init__()
        self.seep_iterations = seep_iterations

    def evolve_population(self, population: List[MetaModel]):
        SELECT_N = 10
        EVOLVE_M_TIMES = 4
        BRANCH_O_TIMES = 2

        actual_select_n = min(len(population), SELECT_N)

        fitness_calculator = AccuracyCalculator()
        dataset = Dataset.get_cifar10()

        population.sort(key=lambda x: x.fitness)
        best_candidates = population[:actual_select_n]
        removed_candidates = population[-actual_select_n:]
        population = population[-actual_select_n:]

        new_candidates = []
        for candidate in best_candidates:
            best_sample = candidate
            for evolution_round in range(EVOLVE_M_TIMES):
                children = [best_sample.produce_child() for _ in range(BRANCH_O_TIMES)]
                for child in children:
                    # TODO: optimization: load base model to speed up with previous weights?
                    child.build_model(dataset.images_shape)
                    child.evaluate(dataset)
                    child.fitness = fitness_calculator.calculate_fitness(child.metrics)
                children.sort(key=lambda x: x.fitness)
                best_sample = children[0]
            new_candidates.append(best_sample)

        return population, new_candidates, removed_candidates

class PyramidStrategy(EvolutionStrategy):
    def __init__(self, seep_iterations):
        super().__init__()
        self.seep_iterations = seep_iterations

    def evolve_population(self, population: List[MetaModel]):
        # each round, take of n*2 candidates, and evolve each other one
        # OR, each round, take off n*2 candidates, and train the remaining ones for 1 number of iterations

        return population, [], []