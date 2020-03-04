from abc import ABC, abstractmethod


class SerialData(ABC):
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def deserialize(self, obj: dict) -> None:
        pass
