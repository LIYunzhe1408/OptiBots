import numpy as np
import timor
from timor.Module import *
from timor.utilities.visualization import animation
# import matplotlib.pyplot as plt
# import itertools
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from timor import ModuleAssembly, ModulesDB
from timor.configuration_search.GA import GA
from timor.utilities.visualization import MeshcatVisualizer, clear_visualizer
import argparse
import pygad



#reachability

def load_modules():
    #TODO - load in modules from a file, or allow a way to define some random links/joints/connectors

    return ModulesDB.from_name('modrob-gen2') 

def populate_hyperparameters(args: dict() = {}):

    #hardcode some default values
    hyperparameters = {
        'population_size': 10,
        'num_generations': 150,
        'num_genes': 6,
        'save_solutions_dir': None
    }

    for k, v in args.items():
        if k in hyperparameters:
            hyperparameters[k] = v

    return hyperparameters


def fitness_function(assembly: ModuleAssembly, ga_instance: pygad.GA, index: int) -> Lexicographic:
    """
    This fitness function returns a lexicographic value, where the
    
    - first value indicates the number of modules in the robot, and
    - the second value the minus of the total mass
    """
    #TODO - include reachability metric and max weight calculations in this fitness function

    #this function is an example function.
    return Lexicographic(len(assembly.module_instances), -assembly.mass)



def main(hyperparameters, visualize = False):

    db = populate_modules()

    if visualize:
        viz = MeshcatVisualizer()
        viz.initViewer()
        db.debug_visualization(viz) 

    ga = GA(db)
    ga_optimizer = ga.optimize(fitness_function=fitness_scalar, hp = hyperparameters, save_best_solutions=False)

    print("The best robot in the initial population had a mass of:", -ga_optimizer.best_solutions_fitness[0], "kg")
    print("The best robot had a mass of:", -ga_optimizer.best_solution()[1], "kg")
    print("The best robot had", len([module for module in ga_optimizer.best_solution()[0] if module != 0]), "modules")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action='store_true')

    args = parser.parse_args()
    hyperparameters = populate_hyperparameters(vars(args))

    main(hyperparameters, visualize = args.visualize)