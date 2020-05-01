from __future__ import annotations
import sys
import numpy as np
from scipy import stats

from EvolutionStrategy import AgingStrategy
from FitnessCalculator import AccuracyCalculator
from Dataset import ImageDataset
import matplotlib.pyplot as plt
import time
from FileManagement import *
from NASModel import MetaModel, print_vars
from SerialData import SerialData
from Hyperparameters import Hyperparameters
from ModelExperiments import *


HERE = os.path.dirname(os.path.abspath(__file__))

'''
TODO:

=HIGH PRIORITY=
- soft vs hard fitness curve for accuracy / time tradeoff
- config determines fitness calculator
- early stopping determined by delta test accuracy

=MEDIUM PRIORITY=
- add researched selection routine
- non-overlap selection for aging selection ***********
- increasing epochs over time?
- dataset shuffler callback shuffle method



=FOR CONSIDERATION=
- should block output be dim reduced after concat?

=EXPERIMENTS=
coorelation between FLOPS vs size vs time
coorelation betweeen accuracy of low filter numbers vs high

'''

'''
pareto sampling
scheduled droppath
cosine annealing
rounded curve for accuracy tradeoff

1. Aging Selection
    1. select a set S candidates from population P
    2. select a set N candidates from S to have children (set M)
    3. mutate all candidates in M
    4. train all children in M
    5. Add all children in M to P
    6. Remove the |M| oldest candidates from P 

2. Tournament Selection
    select two individuals, kill the less fit one, and the more fit one has a child
'''


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


def do_evolution(dir_path: str, num_rounds: int, is_debug: bool = False):
    config_path_name = 'config'
    progress_path_name = 'progress'

    params = Hyperparameters(is_debug)
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
    if is_debug:
        dataset = ImageDataset.get_build_set()
    else:
        dataset = ImageDataset.get_cifar10()

    fitness_calculator = AccuracyCalculator()

    evolution_strategy = AgingStrategy(params.parameters['STRATEGY_SELECTION_SIZE'])
    if params.parameters['STRATEGY'] == 'aging':
        evolution_strategy = AgingStrategy(params.parameters['STRATEGY_SELECTION_SIZE'])

    def handle_new_candidate(new_candidate):
        new_candidate.model_name = 'evo_' + str(time.time())  # this is redone here since all models are initialized within microseconds of eachother for init population
        new_candidate.build_model(dataset.images_shape)
        new_candidate.evaluate(dataset)
        new_candidate.save_model(dir_path)
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

    for r in range(evolution_rounds_to_conduct):
        print(f'Performing evolution round {r}')
        population, new_candidates, removed_candidates = evolution_strategy.evolve_population(population)
        for candidate in new_candidates:
            handle_new_candidate(candidate)
            write_progress()
        for candidate in removed_candidates:
            # candidate.save_model(dir_path)
            candidate.clear_model()

    # for remaining_candidate in population:
    #     remaining_candidate.save_model(dir_path)
    #     remaining_candidate.clear_model()

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

    for candidate in history:
        candidate.plot_model(dir_path)




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

    else:
        num_args = len(sys.argv) - 1
        is_debug = sys.argv[1] == 'debug'
        arg_pos = 1
        if is_debug:
            num_args -= 1
            arg_pos += 1

        if num_args == 1:
            if sys.argv[arg_pos] == 'help':
                help_text()
            if sys.argv[arg_pos] == 'init':
                do_evolution(evo_dir, 0, is_debug)

            if sys.argv[arg_pos] == 'run':
                do_evolution(evo_dir, -1, is_debug)

            if sys.argv[arg_pos] == 'analyze':
                plot_history(evo_dir)

            if sys.argv[arg_pos] == 'test':
                test_accuracy_at_different_train_amounts()

            if sys.argv[arg_pos] == 'test2':
                test_accuracy_at_different_train_amounts_2()

            if sys.argv[arg_pos] == 'test3':
                test_model_mutation()

            if sys.argv[arg_pos] == 'test4':
                train_models_more()

            if sys.argv[arg_pos] == 'test_measure':
                test_accuracy_at_different_train_amounts_analyze()

            if sys.argv[arg_pos] == 'test_measure2':
                test_accuracy_at_different_train_amounts_analyze_2()

            if sys.argv[arg_pos] == 'test_measure3':
                analyze_mutations()

        elif num_args == 2:
            if sys.argv[arg_pos] == 'init':
                provided_path = sys.argv[arg_pos + 1]
                do_evolution(provided_path, 0, is_debug)

            if sys.argv[arg_pos] == 'run':
                number_rounds = int(sys.argv[arg_pos + 1])
                do_evolution(evo_dir, number_rounds, is_debug)

            if sys.argv[arg_pos] == 'analyze':
                provided_path = sys.argv[arg_pos + 1]
                plot_history(provided_path)

        elif num_args == 3:
            if sys.argv[arg_pos] == 'run':
                provided_path = sys.argv[arg_pos + 1]
                number_rounds = int(sys.argv[arg_pos + 2])
                do_evolution(provided_path, number_rounds, is_debug)
