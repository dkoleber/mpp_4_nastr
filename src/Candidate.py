from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
from Dataset import Dataset

class Candidate(ABC):
    def __init__(self):
        self.fitness = 0
        self.age = 0

    @abstractmethod
    def evaluate_fitness(self, dataset: Dataset) -> None:
        pass

    @abstractmethod
    def duplicate(self) -> Candidate:
        pass

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass
