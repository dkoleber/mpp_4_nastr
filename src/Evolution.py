from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from NetworkBuilder import Model
from Dataset import Dataset
from Candidate import Candidate
import matplotlib.pyplot as plt
import time

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
        return population[1:], [new_candidate], [population[0]]


def do_evolution():
    rounds = 1
    population_size = 1

    dataset = Dataset.get_build_set()

    evolution_strategy = AgingStrategy()
    population = [Model() for _ in range(population_size)]
    for index, candidate in enumerate(population):
        print(f'evaluating candidate {index} of initial population')
        candidate.model_name = 'evo_' + str(time.time())
        candidate.populate_with_NASnet_blocks()
        candidate.evaluate_fitness(dataset)

    history = population

    print(len(history))
    for r in range(rounds):
        print(len(history))
        print(f'performing evolution round {r}')
        population, new_candidates, removed_candidates = evolution_strategy.evolve_population(population)
        print(len(history))
        history.extend(new_candidates)
        print(len(new_candidates), len(history))
        for candidate in new_candidates:
            candidate.model_name = 'evo_' + str(time.time())
            candidate.evaluate_fitness(dataset)

    history_fitness = [x.fitness for x in history]
    best_candidate = int(np.argmax(history_fitness))
    return history, best_candidate

def plot_history(history):
    x = [x for x in range(len(history))]
    y = [x.fitness for x in history]

    print(x, y)

    area = np.pi*5

    plt.scatter(x, y, s=area, alpha=0.5)
    plt.title('test_plot')
    plt.xlabel('candidate')
    plt.ylabel('fitness')
    plt.show()

def do_test():
    history, best_candidate_index = do_evolution()
    plot_history(history)


if __name__ == '__main__':
    do_test()