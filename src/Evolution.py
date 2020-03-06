from __future__ import annotations
import sys
import numpy as np
from EvolutionStrategy import AgingStrategy
from FitnessCalculator import AccuracyCalculator
from Dataset import Dataset
import matplotlib.pyplot as plt
import time
from FileManagement import *
from Modelv3 import MetaModel
from SerialData import SerialData
from Hyperparameters import Hyperparameters

HERE = os.path.dirname(os.path.abspath(__file__))

DEBUG = False

class EvolutionProgress(SerialData):
    def __init__(self):
        self.parameters = {
            'population_names': [],
            'history_names': [],
        }

    def serialize(self) -> dict:
        return self.parameters

    def deserialize(self, obj: dict) -> None:
        self.parameters = obj


def do_evolution(dir_path: str, num_rounds: int):
    config_path_name = 'config'
    progress_path_name = 'progress'

    params = Hyperparameters(DEBUG)
    progress = EvolutionProgress()

    config_path = os.path.join(dir_path, f'{config_path_name}.json')
    progress_path = os.path.join(dir_path, f'{progress_path_name}.json')

    # == DESERIALIZE CONFIG AND PROGRESS ==

    if os.path.exists(dir_path):
        if os.path.exists(config_path):
            serialized_params = read_json_from_file(dir_path, config_path_name)
            params.deserialize(serialized_params)
        if os.path.exists(progress_path):
            serialized_progress = read_json_from_file(dir_path, progress_path_name)
            progress.deserialize(serialized_progress)
    else:
        os.makedirs(dir_path)
    if not os.path.exists(config_path):
        serialized_params = params.serialize()
        write_json_to_file(serialized_params, dir_path, config_path_name)
    if not os.path.exists(progress_path):
        serialied_progress = progress.serialize()
        write_json_to_file(serialied_progress, dir_path, progress_path_name)

    # == LOAD ANY PREVIOUS MODELS ==

    history_names = [x for x in os.listdir(dir_path) if '.json' not in x]
    history = []
    population = []
    for name in history_names:
        model = MetaModel.load(dir_path, name)
        if model.hyperparameters == params:
            if name in progress.parameters['history_names']:
                history.append(model)
            if name in progress.parameters['population_names']:
                population.append(model)

    # == CALCULATE THE NUMBER OF ROUND OF INITIALIZATION AND/OR EVOLUTION ==

    actual_num_rounds = num_rounds
    if actual_num_rounds < 0:
        actual_num_rounds = params.parameters['POPULATION_SIZE'] + params.parameters['ROUNDS']
    if len(history) == (params.parameters['POPULATION_SIZE'] + params.parameters['ROUNDS']):
        actual_num_rounds = 0
    init_population_target = params.parameters['POPULATION_SIZE']
    init_population_remaining = init_population_target - len(population)
    init_population_to_conduct = min(init_population_remaining, actual_num_rounds)
    rounds_remaining_after_population = actual_num_rounds - init_population_to_conduct
    evolution_rounds_target = params.parameters['ROUNDS']
    evolution_rounds_remaining = evolution_rounds_target - (len(history) - len(population))
    evolution_rounds_to_conduct = min(evolution_rounds_remaining, rounds_remaining_after_population)
    print()
    print(f'Evaluating {init_population_to_conduct} of initial population ({init_population_remaining} of {init_population_target} remain)')
    print(f'Evaluating {evolution_rounds_to_conduct} evolution rounds ({evolution_rounds_remaining} of {evolution_rounds_target} remain)')
    print()

    if actual_num_rounds == 0:
        return history

    # == LOAD THE DATASET ==

    dataset = None
    if DEBUG:
        dataset = Dataset.get_build_set()
    else:
        dataset = Dataset.get_cifar10()

    fitness_calculator = AccuracyCalculator()

    evolution_strategy = AgingStrategy(params.parameters['STRATEGY_SELECTION_SIZE'])
    if params.parameters['STRATEGY'] == 'aging':
        evolution_strategy = AgingStrategy(params.parameters['STRATEGY_SELECTION_SIZE'])

    def handle_new_candidate(new_candidate):
        new_candidate.build_model(dataset.images_shape)
        new_candidate.evaluate(dataset)
        new_candidate.save_metadata(dir_path)
        new_candidate.fitness = fitness_calculator.calculate_fitness(new_candidate.metrics)
        history.append(new_candidate)

    def write_progress():
        progress.parameters['history_names'] = [x.model_name for x in history]
        progress.parameters['population_names'] = [x.model_name for x in population]
        write_serialized_progress = progress.serialize()
        write_json_to_file(write_serialized_progress, dir_path, progress_path_name)

    new_population = [MetaModel(params) for _ in range(init_population_to_conduct)]
    for index, candidate in enumerate(new_population):
        print(f'Evaluating candidate {index} of initial population')
        candidate.populate_with_nasnet_metacells()
        handle_new_candidate(candidate)
        population.append(candidate)
        write_progress()

    # print(history[-1].model_data.final_dense_2.get_weights())

    for r in range(evolution_rounds_to_conduct):
        print(f'Performing evolution round {r}')
        population, new_candidates, removed_candidates = evolution_strategy.evolve_population(population)
        for candidate in new_candidates:
            handle_new_candidate(candidate)
            write_progress()
        for candidate in removed_candidates:
            candidate.clear_graph()

    # print(history[-1].model_data.final_dense_2.get_weights())


    # tc = MetaModel.load(dir_path, history[-1].model_name, True)
    # tc.model_name = 'evo_' + str(time.time())
    # tc.evaluate(dataset)
    # tc.save_graph(dir_path)
    # tc.save_metadata(dir_path)

    return history


def plot_history(dir_path: str):
    config_path_name = 'config'
    progress_path_name = 'progress'

    params = Hyperparameters()
    progress = EvolutionProgress()

    config_path = os.path.join(dir_path, f'{config_path_name}.json')
    progress_path = os.path.join(dir_path, f'{progress_path_name}.json')

    if os.path.exists(dir_path):
        if os.path.exists(config_path):
            serialized_params = read_json_from_file(dir_path, config_path_name)
            params.deserialize(serialized_params)
        else:
            print(f'Could not find \"{config_path}\" in \"{dir_path}\"')
            return
        if os.path.exists(progress_path):
            serialized_progress = read_json_from_file(dir_path, progress_path_name)
            progress.deserialize(serialized_progress)
        else:
            print(f'Could not find \"{progress_path}\" in \"{dir_path}\"')
            return
    else:
        print(f'\"{dir_path}\" is not a valid directory')
        return

    history_names = [x for x in os.listdir(dir_path) if '.json' not in x]
    history = []
    for name in history_names:
        model = MetaModel.load(dir_path, name)
        if model.hyperparameters == params:
            if name in progress.parameters['history_names']:
                history.append(model)

    fitness_calculator = AccuracyCalculator()

    for candidate in history:
        candidate.fitness = fitness_calculator.calculate_fitness(candidate.metrics)

    x = [x for x in range(len(history))]
    y = [x.fitness for x in history]

    area = np.pi*8

    plt.scatter(x, y, s=area, alpha=0.5)
    plt.title('test_plot')
    plt.xlabel('candidate')
    plt.ylabel('fitness')
    plt.show()


if __name__ == '__main__':

    def help_text():
        print(f'python Evolution.py init [<path>]')
        print(f'    <path> defaults to ./evolution')
        print(f'    Initializes an evolution configuration at the <path>.')
        print()
        print(f'python Evolution.py run [<path> [<number>]]')
        print(f'    <path> defaults to ./evolution')
        print(f'    <number> defaults to -1')
        print(f'    Targets directory <path> to contain candidates and metadata.')
        print(f'    Runs <number> rounds of evaluation. If <number> < 0, then runs entire population and evolution.')
        print(f'    Evaluation first includes the initial population, then any candidates produced during evolution.')
        print()
        print(f'python Evolution.py analyze [<path>]')
        print(f'    <path> defaults to ./evolution')
        print(f'    Analyzes all candidates within the <path> and produces a chart.')
        print()

    if len(sys.argv) == 1:
        help_text()

    elif len(sys.argv) == 2:
        if sys.argv[1] == 'help':
            help_text()
        if sys.argv[1] == 'init':
            do_evolution(evo_dir, 0)

        if sys.argv[1] == 'run':
            do_evolution(evo_dir, -1)

        if sys.argv[1] == 'analyze':
            plot_history(evo_dir)

    elif len(sys.argv) == 3:
        if sys.argv[1] == 'init':
            provided_path = sys.argv[2]
            do_evolution(provided_path, 0)

        if sys.argv[1] == 'run':
            number_rounds = int(sys.argv[2])
            do_evolution(evo_dir, number_rounds)

        if sys.argv[1] == 'analyze':
            provided_path = sys.argv[2]
            plot_history(provided_path)

    elif len(sys.argv) == 4:
        if sys.argv[1] == 'run':
            provided_path = sys.argv[2]
            number_rounds = int(sys.argv[3])
            do_evolution(provided_path, number_rounds)
