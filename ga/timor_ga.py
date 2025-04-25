import numpy as np
import timor
import argparse
import sys
import pygad
import os
import ray
import time
from util import *
from timor.Module import *
from timor.utilities.visualization import animation
# import matplotlib.pyplot as plt
# import itertools
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from timor import ModuleAssembly, ModulesDB
from timor.Bodies import Body, Connector, Gender
#from timor.configuration_search.GA import GA
from GA import GA
from timor.utilities.visualization import MeshcatVisualizer, clear_visualizer
from timor.utilities.dtypes import Lexicographic
from timor.utilities.transformation import Transformation
from timor.utilities.spatial import rotX, rotY, rotZ
from timor.Module import AtomicModule, ModulesDB, ModuleHeader
from timor.Joints import Joint
from timor.Geometry import Box, ComposedGeometry, Cylinder, Sphere, Mesh
from timor.task.Obstacle import Obstacle
from timor.task.Task import Task, TaskHeader

from reachability import Reachability
from reachability_with_weight import Reachability_with_weight
from reachability_parallel import Reachability, generate_tasks
from generate_module import create_i_links, create_eef, create_revolute_joint, generate_i_links

project_root = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(project_root)

from random_env_generation.random_env_generation import plot_random_cuboids, plot_random_cuboids_with_reachability, plot_reachability_interactive

how_many_times_to_split_angle_range = 30
world_resolution = 0.01
world_dimension = [1.00, 1.00, 1.00]
num_threads = 5

ga_args = {
    'parallel_processing': ['process', 30],
}
db = None

improved_hyperparameters = {
    'num_generations': 200,
    'num_parents_mating': 15,
    'population_size': 30,
    'num_genes': 14,
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
NUM2ID = {}
our_hyperparameters = {
    'population_size': 20,
    'num_generations': 150,
    'num_genes': 14,
    'keep_parents': 1,
    #'gene_space': constrained_gene_space(),
    'save_solutions_dir': None
}
#6659
NUM_SAMPLE = 7000 #100000 

def generate_tasks(n, num_obstacles=3):
    tasks = []
    for i in range(n):
        cuboid_data_list = plot_random_cuboids(num_obstacles)
        header = TaskHeader(
            ID='Random Obstacles Generation v: ' + str(i) ,
            tags=['Capstone', 'demo'],
            date=datetime.datetime(2024, 10, 28),
            author=['Jonas Li, Jae Won Kim'],
            email=['liyunzhe.jonas@berkeley.edu, jaewon_kim@berkeley.edu'],
            affiliation=['UC Berkeley']
        )
        with open("tasks.txt", "w") as f:
            for item in cuboid_data_list:
                f.write(f"{item}\n")
        box = []
        for idx, info in enumerate(cuboid_data_list):
            size, displacement = info["size"], info["origin"]
            box.append(Obstacle(ID=str(idx), 
                        collision=Box(
                            dict(x=size['x'], y=size['y'], z=size['z']),  # Size
                            pose=Transformation.from_translation([displacement['x'] + size['x'] / 2, displacement['y'] + size['y'] / 2, displacement['z'] + size['z'] / 2])
                        )))
        task = Task(header, obstacles=[i for i in box])
        tasks.append(task)
    return tasks

TASKS = generate_tasks(3, 20)
  # each element on a new line
def num_incorrect_connections(moduleAssembly):
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
    num_incorrect_connections = 0
    modules = list(moduleAssembly.original_module_ids)
    if not modules:
        return True
    
    # Helper function to extract joint names from a link
    def extract_joints_from_link(link):
        # Handle different link naming patterns
        if "-to-" in link:
            parts = link.split("-to-")
            # Extract parts before 'to' and after 'to' (before any dash)
            first_joint = parts[0]
            second_joint = parts[1].split("-")[0] if "-" in parts[1] else parts[1]
            return first_joint, second_joint
        return None, None
    
    # Helper function to check if a module is a joint
    def is_joint(module):
        return "-to-" not in module
        #return "joint" in module or module == "eef"
        #return "J" in module or module == "eef"
    
    # Helper function to check if a module is a link
    def is_link(module):
        return "-to-" in module
        #return "-" in module and "to" in module
        #return "i" in module or "l" in module
    
    # Check alternating pattern and correct linkage
    for i in range(len(modules) - 1):
        current = modules[i]
        next_module = modules[i + 1]
        if next_module == 'eef':
            continue

        # Check alternating pattern
        if is_joint(current) and is_joint(next_module):
            # Two joints in a row - should have a link between them
            #return (current, next_module)
            num_incorrect_connections += 1
            #return False
        
        if is_link(current) and is_link(next_module):
            # Two links in a row - invalid
            #return (current, next_module)
            num_incorrect_connections += 1
            #return False
        
        # If current is a joint and next is a link, verify the link connects to this joint
        if is_joint(current) and is_link(next_module):
            first_joint, second_joint = extract_joints_from_link(next_module)
            joint_base_name = current #.replace("_rev_joint", "").replace("_joint", "")
            
            if not (first_joint in joint_base_name):
                #return (first_joint, joint_base_name)
                num_incorrect_connections += 1
                #return False
        
        # If current is a link and next is a joint, verify the link connects to the next joint
        if is_link(current) and is_joint(next_module):
            first_joint, second_joint = extract_joints_from_link(current)
            joint_base_name = next_module #.replace("_rev_joint", "").replace("_joint", "")
            
            if not (second_joint in joint_base_name):
                #return (joint_base_name, second_joint)
                num_incorrect_connections += 1
                #return False
    
    return num_incorrect_connections

def cost_of_robot(assembly: ModuleAssembly):
    modules = list(assembly.original_module_ids)
    total_cost = 0
    for module in modules:
        # 540_base cost is excluded since it is essential for every robot
        if module == "540_joint":
            total_cost += 473.96
        elif module == "430_joint":
            total_cost += 396.79
        elif module == "330_joint":
            total_cost += 99.11
        else:
            continue

    return total_cost
def fitness_function(assembly: ModuleAssembly, ga_instance: pygad.GA, index: int) -> Lexicographic:
    """
    This fitness function returns a lexicographic value, where the
    
    - first value indicates the number of modules in the robot, and
    - the second value the minus of the total mass
    """
    #TODO - include reachability metric and max weight calculations in this fitness function

    robot = assembly.to_pin_robot()
    num_i = num_incorrect_connections(assembly)
    cost = cost_of_robot(assembly)
    # if robot.has_self_collision():
    #     return Lexicographic(-10000,-100, -10)
    if num_i != 0:
        return Lexicographic(-num_i/assembly.nModules, -10000, -10 * assembly.nJoints, -10 * assembly.nModules, -1000)
    #reachability = Reachability(robot=assembly.to_pin_robot(), angle_interval=how_many_times_to_split_angle_range, world_resolution=world_resolution)

    #print(len(TASKS))
    reachability_score = 0
    for task in TASKS:
        reachability = Reachability_with_weight(robot=robot, angle_interval=how_many_times_to_split_angle_range, world_resolution=world_resolution, task=task)
        # reachability = Reachability(robot_modules=assembly.original_module_ids, task=task, db=db)
        # reachability.reachability_random_sample_actors(num_samples = 5000, num_actors = 4, batch_size = 1000)
        # reachability_score += reachability.find_reachibility_percentage(voxel_size = 0.1)
        valid_poses = reachability.reachability_random_sample(num_samples = NUM_SAMPLE, mass=0.7)
        if not valid_poses:
            reachability_score += 0
            return Lexicographic(0, -cost, -10 * assembly.nJoints, -10 * assembly.nModules, -1000)
        else:
            reachable = valid_poses
            reachable = np.array([list(pt) for pt in reachable])
            reachability_score += reachability.find_reachibility_percentage(valid_pose=reachable)
    num_links = assembly.nModules - assembly.nJoints - 1
    return Lexicographic(reachability_score * 100 / len(TASKS), -cost, -10 * assembly.nJoints, -10 * assembly.nModules, -assembly.mass * 100)

# def fitness_function(assembly: ModuleAssembly, ga_instance: pygad.GA, index: int) -> Lexicographic:
#     """
#     This fitness function returns a lexicographic value, where the
    
#     - first value indicates the reachability score of the robot
#     - second value the number of joints (DOF) (negative to incentivise lower values)
#     - third value the number of modules (negative to incentivise lower values)
#     """
#     robot = assembly.to_pin_robot()
#     reachability_score = 0
#     for task in TASKS: # TASKS is defined and initiallized at the beginning of the script
#         reachability = Reachability_with_weight(robot, angle_interval, world_resolution, task)
#         reachability_score += reachability.reachability_random_sample(num_samples, mass)
    
#     return Lexicographic(reachability_score * 100 / len(TASKS), -10 * assembly.nJoints, -10 * assembly.nModules)


def on_generation(ga):
    print("Last generation fitness: ", ga.last_generation_fitness)
    print("Best Fitness: ", ga.best_solutions_fitness)
    with open("generation_fitness.txt", "a") as file:  # Open file in append mode
        file.write(f"Last generation fitness: {ga.last_generation_fitness}\n")
    
    #num2id = {v: k for k, v in ga.id2num.items()}
    solution_module = [] 
    #print("Num2ID: ", NUM2ID)
    for solution in ga.best_solutions:
        solution_module.append([NUM2ID[num] for num in solution])
    with open("best_solution.txt", "a") as file:  # Open file in append mode
        file.write(f"Best fitness: {ga.best_solutions_fitness}\n")
        file.write(f"Best solution: {solution_module}\n\n")


# def naive_optimize(db, baseNames, baseLinkNames, jointNames, linkNames, eefNames, startDOF, endDOF):
#     """
#     Performs a brute-force optimization to find the configuration with the highest fitness score.
#     Iterates over possible configurations (base, joints, links, degrees of freedom, end-effector) and evaluates their fitness.
#     Returns the configuration tuple with the highest fitness score."
#     """

#     joint_link_pairs = list(itertools.product(jointNames, linkNames))
#     maximLexicograph = Lexicographic(float("-inf"),float("-inf"))
#     max_config = ()
#     current_modules_list = []
#     for baseName in baseNames:
#         current_modules_list.append(baseName)
#         for baseLinkName in baseLinkNames:
#             current_modules_list.append(baseLinkName)
#             for dof in range(startDOF, endDOF+1):
#                 for dofSection in itertools.product(joint_link_pairs, repeat=dof):
#                     dofSectionList = sum(list([list(tup) for tup in dofSection]), [])
#                     for eefName in eefNames:
#                         configuration_tuple = tuple(current_modules_list + dofSectionList + [eefName])
#                         current_Lexicograph = fitness_function(ModuleAssembly.from_serial_modules(db, configuration_tuple), None, 1)
#                         if   current_Lexicograph > maximLexicograph:
#                             max_config = configuration_tuple
#                             maximLexicograph = current_Lexicograph

                    
#             current_modules_list.pop()

#         current_modules_list.pop()
#     return max_config

def naive_optimize(db, baseNames, jointNames, base_or_joint_to_compatible_links,
                   joint_to_link_and_next_joint, eefNames, startDOF, endDOF):
    """
    One-pass DFS robot builder that finds the best configuration in a given DOF range.
    Evaluates only when DOF ∈ [startDOF, endDOF], but continues growing up to endDOF.
    """
    maxLex = Lexicographic(float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_config = ()

    for base in baseNames:
        base_links = base_or_joint_to_compatible_links.get(base, [])
        for base_link in base_links:
            try:
                _, _, first_joint, *_ = base_link.split("-")
            except ValueError:
                print(f"Skipping malformed base link: {base_link}")
                continue

            def dfs(depth, config_so_far, current_joint):
                # print(config_so_far)
                nonlocal maxLex, best_config

                # Base case: do not exceed endDOF
                if depth > endDOF:
                    return

                # Evaluate configurations at acceptable DOF range
                if depth >= startDOF:
                    print("depth: ", depth, "config_so_far: ", config_so_far)
                    for eef in eefNames:
                        full_config = tuple(config_so_far + [eef])
                        try:
                            robot = ModuleAssembly.from_serial_modules(db, full_config)
                            score = fitness_function(robot, None, 1)
                            if score > maxLex:
                                maxLex = score
                                best_config = full_config
                        except Exception as e:
                            print(f"Assembly failed: {e}")

                # Try all valid (link, next_joint) continuations
                for link, next_joint in joint_to_link_and_next_joint.get(current_joint, []):
                    dfs(depth + 1, config_so_far + [current_joint, link], next_joint)

            # Begin DFS from each base → base_link → first_joint
            dfs(1, [base, base_link], first_joint)

    return best_config
    



def optimize(db, hyperparameters):
    print("Optimize start")
    ga = GA(db, custom_hp=hyperparameters)
    print("GA instance created")
    num2id = {v: k for k, v in ga.id2num.items()} 
    #print(num2id)
    global NUM2ID
    NUM2ID = num2id
    #print("Num2ID: ", NUM2ID)
    #print("Initial Population: ", ga._get_initial_population())

    ga_optimizer = ga.optimize(fitness_function=fitness_function, selection_type= "tournament", save_best_solutions=True, parallel_processing=("thread", 36), on_generation=on_generation)
    
    module_ids = [num2id[num] for num in ga_optimizer.best_solution()[0]]

    return (module_ids, ga)

def filter(module):
    if "l" in module.id:
        return False
    return True


def main(hyperparameters = None, visualize = False):



    eef = create_eef()
    r_430_joint = create_revolute_joint("assets/430_joint/430_joint/urdf/430_joint.urdf")
    r_330_joint = create_revolute_joint("assets/330_joint/330_joint/urdf/330_joint.urdf")
    r_540_joint = create_revolute_joint("assets/540_joint/540_joint/urdf/540_joint.zip.urdf")
    r_540_base = create_revolute_joint("assets/540_base/540_base/urdf/540_base.urdf")

    generated_links = generate_i_links(r_540_base, [r_330_joint, r_430_joint, r_540_joint])
    # Create database
    global db
    db = ModulesDB()
    db.add(r_330_joint)
    db.add(r_430_joint)
    db.add(r_540_joint)
    db.add(r_540_base)

    db = db.union(eef)
    db = db.union(generated_links)
    viz = db.debug_visualization()

    
    #baseNames = [r_4310_base.id]
    # print(baseNames)
    #eefNames = list(eef.by_id.keys())
    # print(eefNames)

    # from timor.utilities.file_locations import get_module_db_files
    # modules_file = get_module_db_files('geometric_primitive_modules')
    # db = ModulesDB.from_json_file(modules_file)
    # db2 = db.filter(filter)
    # print(db2.all_module_names)
    optimized_results, ga = optimize(db, our_hyperparameters)

    # # when adding the other joints
    # # r_4305_joint.id
    # jointNames = [ r_4310_joint.id]
    # # print(jointNames)

    # baseLinkNames = list(baseto4310_links.by_id.keys())
    # # print(baseLinkNames)

    # # when adding the other links
    # #list(r4310to4305_links.by_id.keys()) +  
    # linkNames = list(r4310to4310_links.by_id.keys())
    # # print(linkNames)

    # ## NAIVE SOLUTION
    # naive_results = naive_optimize(db, baseNames, baseLinkNames, jointNames, linkNames, eefNames, 4, 4)
    # print(naive_results)
    
    # # GA SOLUTION:
    # optimized_results = optimize(db, our_hyperparameters)

    print(optimized_results)

    best_robot_config = naive_optimize(
        db=db,
        baseNames=baseNames,
        jointNames=jointNames,
        base_or_joint_to_compatible_links=base_or_joint_to_compatible_links,
        joint_to_link_and_next_joint=joint_to_link_and_next_joint,
        eefNames=eefNames,
        startDOF=2,  # desired minimum DOF
        endDOF=2     # desired maximum DOF
    )
    print(best_robot_config)



# ('base', 'i_45', 'J1', 'i_15', 'J2', 'i_45', 'J2', 'i_45', 'J1', 'i_15', 'J2', 'i_30', 'J2', 'i_45', 'J2', 'eef')
# ('540_base', '540_base-to-330_joint-0.1-1-S', '330_joint', '330_joint-to-540_joint-0.1-3-W', '540_joint', '540_joint-to-430_joint-0.3-0-E', '430_joint', '430_joint-to-540_joint-0.15-0-W', '540_joint', '540_joint-to-330_joint-0.2-1-E', '330_joint', '330_joint-to-430_joint-0.2-0-N', 'EMPTY', 'eef'], ['540_base', '540_base-to-330_joint-0.1-1-S', '330_joint', '330_joint-to-540_joint-0.1-3-W', '540_joint', '540_joint-to-430_joint-0.3-0-E', '430_joint', '430_joint-to-540_joint-0.15-0-W', '540_joint', '540_joint-to-330_joint-0.2-1-E', '330_joint', '330_joint-to-430_joint-0.2-0-N', 'EMPTY', 'eef')
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--visualize", action='store_true')

    # args = parser.parse_args()
    # hyperparameters = populate_hyperparameters(vars(args))
    # 24% : ['base_rev_joint', 'EMPTY', 'baseto4310-0.45', 'motor4310_rev_joint', 'r4310to4305-0.3', 'motor4305_rev_joint', 'r4310to4305-0.3', 'motor4310_rev_joint', 'eef']
    main()