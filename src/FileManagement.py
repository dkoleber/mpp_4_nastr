import os
import json
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

evo_dir = HERE / '../evolution/'
res_dir = HERE / '../res/'
tensorboard_dir = HERE / '../tensorboard/'
model_save_dir = HERE / '../models/'
task_dataset_dir = HERE / '../task_datasets/'
fig_dir = res_dir / 'figures/'

dirs = [evo_dir, res_dir, tensorboard_dir, model_save_dir, task_dataset_dir, fig_dir]


def write_json_to_file(obj: dict, dir_path: str, name: str) -> None:
    with open(os.path.join(dir_path, name + '.json'), 'w') as fl:
        json.dump(obj, fl, indent=4)


def read_json_from_file(dir_path: str, name: str) -> dict:
    with open(os.path.join(dir_path, name + '.json'), 'r') as fl:
        serialized = json.load(fl)
    return serialized


for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
