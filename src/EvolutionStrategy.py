from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from Modelv2 import MetaModel


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
        new_candidate = sampled_candidates[best_candidate_index].duplicate()
        new_candidate.mutate()
        population.append(new_candidate)
        return population[1:], [new_candidate], [population[0]]