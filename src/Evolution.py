from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

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


def do_evolution():
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
