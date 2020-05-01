from __future__ import annotations

from abc import ABC, abstractmethod
import os

HERE = os.path.dirname(os.path.abspath(__file__))

from FileManagement import *


class SerialData(ABC):
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def deserialize(self, obj: dict) -> None:
        pass

    @staticmethod
    def write_serial_data_to_json(data: SerialData, dir_path: str, name: str) -> None:
        serialized = data.serialize()
        write_json_to_file(serialized, dir_path, name)

    @staticmethod
    def load_serial_data_from_json(dir_path: str, name: str) -> dict:
        serialized = read_json_from_file(dir_path, name)
        return serialized