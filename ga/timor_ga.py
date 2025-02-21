import numpy as np
import timor
import argparse
import pygad
import os
from util import *
from timor.Module import *
from timor.utilities.visualization import animation
# import matplotlib.pyplot as plt
# import itertools
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from timor import ModuleAssembly, ModulesDB
from timor.Bodies import Body, Connector, Gender
from timor.configuration_search.GA import GA
from timor.utilities.visualization import MeshcatVisualizer, clear_visualizer
from timor.utilities.dtypes import Lexicographic
from timor.utilities.transformation import Transformation
from timor.utilities.spatial import rotX, rotY, rotZ
from timor.Module import AtomicModule, ModulesDB, ModuleHeader
from timor.Joints import Joint
from timor.Geometry import Box, ComposedGeometry, Cylinder, Sphere, Mesh

from reachability import Reachability
from generate_module import create_i_links, create_eef, create_revolute_joint

how_many_times_to_split_angle_range = 30
world_resolution = 0.01
world_dimension = [1.00, 1.00, 1.00]
num_threads = 5
our_hyperparameters = {
    'population_size': 10,
    'num_generations': 50,
    'num_genes': 6,
    'save_solutions_dir': None
}

def fitness_function(assembly: ModuleAssembly, ga_instance: pygad.GA, index: int) -> Lexicographic:
    """
    This fitness function returns a lexicographic value, where the
    
    - first value indicates the number of modules in the robot, and
    - the second value the minus of the total mass
    """
    #TODO - include reachability metric and max weight calculations in this fitness function

    # if assembly.nJoints != 3:
    #     return -1000
    reachability = Reachability(robot=assembly.to_pin_robot(), angle_interval=how_many_times_to_split_angle_range, world_resolution=world_resolution)
    valid_poses = reachability.reachability_random_sample(num_samples = 100000)
    reachable_space = reachability.find_reachibility_percentage(world_dim=world_dimension, world_res=world_resolution)

    return reachable_space

def optimize(db, hyperparameters):
    ga = GA(db, hyperparameters) 
    ga_optimizer = ga.optimize(fitness_function=fitness_function, save_best_solutions=True)
    num2id = {v: k for k, v in ga.id2num.items()} 
    module_ids = [num2id[num] for num in ga_optimizer.best_solution()[0]]

    return module_ids

def main(hyperparameters = None, visualize = False):

    # Base and joint
    r_4310_base = create_revolute_joint("assets/Assem_4310_BASE/Assem_4310_BASE/urdf/Assem_4310_BASE.urdf")
    r_4305_joint = create_revolute_joint("assets/Assem_4305_JOINT/Assem_4305_JOINT/urdf/Assem_4305_JOINT.urdf")
    r_4310_joint = create_revolute_joint("assets/Assem_4310_JOINT/Assem_4310_JOINT/urdf/Assem_4310_JOINT.urdf")

    # Links
    baseto4310_links = create_i_links(rod_name="baseto4310")
    r4310to4305_links = create_i_links(rod_name="r4310to4305")
    r4310to4310_links = create_i_links(rod_name="r4310to4310")

    eef = create_eef()

    # Create database
    db = ModulesDB()
    db.add(r_4310_base)
    db.add(r_4310_joint)
    db.add(r_4305_joint)
    db = db.union(baseto4310_links)
    db = db.union(r4310to4305_links)
    db = db.union(r4310to4310_links)
    db = db.union(eef)
    viz = db.debug_visualization()

    optimized_results = optimize(db, our_hyperparameters)

    print(optimized_results)





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--visualize", action='store_true')

    # args = parser.parse_args()
    # hyperparameters = populate_hyperparameters(vars(args))

    main()