from abc import ABC, abstractmethod

from Metrics import Metrics


class FitnessCalculator(ABC):
    @abstractmethod
    def calculate_fitness(self, metrics: Metrics) -> float:
        pass


class AccuracyCalculator(FitnessCalculator):
    def calculate_fitness(self, metrics: Metrics, evaluation_index: int = -1) -> float:
        return metrics.metrics['accuracy'][evaluation_index]
