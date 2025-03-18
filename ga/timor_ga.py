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
    'population_size': 40,
    'num_generations': 200,
    'num_genes': 20,
    'save_solutions_dir': None
}
db = None

improved_hyperparameters = {
    'num_generations': 200,
    'num_parents_mating': 15,
    'population_size': 50,
    'num_genes': 20,
    'keep_parents': 1,
    # 'crossover_type': "uniform",
    # 'mutation_type': "adaptive",
    # 'mutation_percent_genes': [10, 20],
    # 'init_range_low': -4,
    # 'init_range_high': 4
}
# improved_hyperparameters = {
#     'num_generations': 1000,           # More generations to find better solutions
#     'num_parents_mating': 20,          # Slightly more parents
#     'population_size': 100,            # Larger population
#     'num_genes': 9,
#     'parent_selection_type': "rank",   # Rank selection often works better
#     'keep_parents': 5,                 # Keep more good solutions
#     'crossover_type': "two_points",    # Try different crossover
#     'mutation_type': "adaptive",
#     'mutation_percent_genes': [5, 15], # Less aggressive mutation
#     'init_range_low': -10,             # Wider initial range
#     'init_range_high': 10
# }

NUM_SAMPLE = 100000
def is_valid_module_order(moduleAssembly):
    """
    Checks if a list of modules is ordered correctly according to these rules:
    1. Links must be placed between joints they connect
    2. Links cannot connect to other links
    3. Links must connect to joints mentioned in their name
    4. The sequence should alternate between joints and links, ending with a joint or eef
    
    Args:
        modules (tuple or list): A sequence of module names as strings
        
    Returns:
        bool: True if the order is valid, False otherwise
    """
    modules = list(moduleAssembly.original_module_ids)
    if not modules:
        return True
    
    # Helper function to extract joint names from a link
    def extract_joints_from_link(link):
        # Handle different link naming patterns
        if "to" in link:
            parts = link.split("to")
            # Extract parts before 'to' and after 'to' (before any dash)
            first_joint = parts[0]
            second_joint = parts[1].split("-")[0] if "-" in parts[1] else parts[1]
            return first_joint, second_joint
        return None, None
    
    # Helper function to check if a module is a joint
    def is_joint(module):
        return "joint" in module or module == "eef"
    
    # Helper function to check if a module is a link
    def is_link(module):
        return "-" in module and "to" in module
    
    # Check alternating pattern and correct linkage
    for i in range(len(modules) - 2):
        current = modules[i]
        next_module = modules[i + 1]
        
        # Check alternating pattern
        if is_joint(current) and is_joint(next_module):
            # Two joints in a row - should have a link between them
            return (current, next_module)
            return False
        
        if is_link(current) and is_link(next_module):
            # Two links in a row - invalid
            return (current, next_module)
            return False
        
        # If current is a joint and next is a link, verify the link connects to this joint
        if is_joint(current) and is_link(next_module):
            first_joint, second_joint = extract_joints_from_link(next_module)
            joint_base_name = current.replace("_rev_joint", "").replace("_joint", "")
            
            if not (first_joint in joint_base_name):
                #return (first_joint, joint_base_name)
                return False
        
        # If current is a link and next is a joint, verify the link connects to the next joint
        if is_link(current) and is_joint(next_module):
            first_joint, second_joint = extract_joints_from_link(current)
            joint_base_name = next_module.replace("_rev_joint", "").replace("_joint", "")
            
            if not (second_joint in joint_base_name):
                #return (joint_base_name, second_joint)
                return False
    
    return True
    
def fitness_function(assembly: ModuleAssembly, ga_instance: pygad.GA, index: int) -> Lexicographic:
    """
    This fitness function returns a lexicographic value, where the
    
    - first value indicates the number of modules in the robot, and
    - the second value the minus of the total mass
    """
    #TODO - include reachability metric and max weight calculations in this fitness function
    robot = assembly.to_pin_robot()
    if robot.has_self_collision() or assembly.nJoints <= 4 or (not is_valid_module_order(assembly)):
        return Lexicographic(0,-10)
    reachability = Reachability(robot=assembly.to_pin_robot(), angle_interval=how_many_times_to_split_angle_range, world_resolution=world_resolution)
    valid_poses = reachability.reachability_random_sample(num_samples = NUM_SAMPLE)
    reachable, manipulability =  zip(*valid_poses)
    reachable = np.array([list(pt) for pt in reachable])
    reachability_score = reachability.find_reachibility_percentage(valid_pose=reachable)
    #reachability_score = len(valid_poses)/NUM_SAMPLE
    num_links = assembly.nModules - assembly.nJoints - 1
    return Lexicographic(reachability_score, -assembly.nJoints)



def naive_optimize(db, baseNames, baseLinkNames, jointNames, linkNames, eefNames, startDOF, endDOF):
    """
    Performs a brute-force optimization to find the configuration with the highest fitness score.
    Iterates over possible configurations (base, joints, links, degrees of freedom, end-effector) and evaluates their fitness.
    Returns the configuration tuple with the highest fitness score."
    """

    joint_link_pairs = list(itertools.product(jointNames, linkNames))
    maximLexicograph = Lexicographic(float("-inf"),float("-inf"))
    max_config = ()
    current_modules_list = []
    for baseName in baseNames:
        current_modules_list.append(baseName)
        for baseLinkName in baseLinkNames:
            current_modules_list.append(baseLinkName)
            for dof in range(startDOF, endDOF+1):
                for dofSection in itertools.product(joint_link_pairs, repeat=dof):
                    dofSectionList = sum(list([list(tup) for tup in dofSection]), [])
                    for eefName in eefNames:
                        configuration_tuple = tuple(current_modules_list + dofSectionList + [eefName])
                        current_Lexicograph = fitness_function(ModuleAssembly.from_serial_modules(db, configuration_tuple), None, 1)
                        if   current_Lexicograph > maximLexicograph:
                            max_config = configuration_tuple
                            maximLexicograph = current_Lexicograph

                    
            current_modules_list.pop()

        current_modules_list.pop()
    return max_config
    



def optimize(db, hyperparameters):
    ga = GA(db, custom_hp=hyperparameters) 
    ga_optimizer = ga.optimize(fitness_function=fitness_function, selection_type= "tournament", save_best_solutions=True)
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
    global db
    db = ModulesDB()
    db.add(r_4310_base)
    db.add(r_4310_joint)
    #db.add(r_4305_joint)
    db = db.union(baseto4310_links)
    #db = db.union(r4310to4305_links)
    db = db.union(r4310to4310_links)
    db = db.union(eef)
    viz = db.debug_visualization()
    # print(db)
    # print(db.all_joints)
    # print(db.all_connectors)
    # print(db.by_name)
    # print(db.by_id)
    # print(db.by_id)
    # print(list(baseto4310_links.by_id.keys()))
    
    baseNames = [r_4310_base.id]
    # print(baseNames)
    eefNames = list(eef.by_id.keys())
    # print(eefNames)


    # when adding the other joints
    # r_4305_joint.id
    jointNames = [ r_4310_joint.id]
    # print(jointNames)

    baseLinkNames = list(baseto4310_links.by_id.keys())
    # print(baseLinkNames)

    # when adding the other links
    #list(r4310to4305_links.by_id.keys()) +  
    linkNames = list(r4310to4310_links.by_id.keys())
    # print(linkNames)

    ## NAIVE SOLUTION
    naive_results = naive_optimize(db, baseNames, baseLinkNames, jointNames, linkNames, eefNames, 4, 4)
    print(naive_results)
    
    # GA SOLUTION:
    optimized_results = optimize(db, our_hyperparameters)

    print(optimized_results)





if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--visualize", action='store_true')

    # args = parser.parse_args()
    # hyperparameters = populate_hyperparameters(vars(args))
    # 24% : ['base_rev_joint', 'EMPTY', 'baseto4310-0.45', 'motor4310_rev_joint', 'r4310to4305-0.3', 'motor4305_rev_joint', 'r4310to4305-0.3', 'motor4310_rev_joint', 'eef']
    main()