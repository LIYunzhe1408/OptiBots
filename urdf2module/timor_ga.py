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
from timor.utilities.dtypes import Lexicographic
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


def reachability_metric(assembly: ModuleAssembly):

    def specific_pose_valid(robot, theta, task = None) -> bool:
        """
        Given a robot at a specific pose, evaluate whether it is a valid pose (contains no collision).
        TODO: reach target pose from the previous pose

        Args:
            robot (timor.Robot.PinRobot): robot
            theta (list[float]): joint angles configuration
            task (timor.task.Task, optional): environment with obstacles. Defaults to None.

        Returns:
            bool: (for now) True if robot has no collision, False otherwise 
        """		
        # perform FK on the theta list
        g = robot.fk(theta, visual = True, collision = True)

        self_collisions = robot.has_self_collision()
        collisions = False if task is None else robot.has_collisions(task, safety_margin=0) # TODO may need to alter safety margin

        return not (collisions or self_collisions)

    def find_all_valid_poses(robot, angle_interval=100, world_res=0.01, task = None) -> list[tuple[float, float, float]]:
        """
        Finds all the valid poses that are reachable by iterating through the robot's joint space.

        Args:
            robot (timor.Robot.PinRobot): our robot
            angle_interval (int): how many times do we split the range of angles from 0 to 2pi
            world_res (float): used to round the valid poses (argument should look like 0.01m)
            task (timor.task.Task, optional): task containing obstacles. Defaults to None.

        Returns:
            list[tuple[float, float, float]]: a list containing all valid poses x, y, z in the world space
        """
        # Generate all possible combinations of joint angles
        num_joints = len(robot.joints)
        angles = np.linspace(0, 2 * np.pi, angle_interval)
        all_angles_combo = itertools.product(angles, repeat=num_joints)
        
        # To store all valid poses, multiple joint combos may result in the same end-effector position
        valid_poses = set()
        rounding = int(-math.log10(world_res)) # 0.01 resolution corresponds to rounding a number to 2 decimal places
        
        for theta in all_angles_combo:
            # print("Testing theta comb: ", theta) # for testing
            
            if (specific_pose_valid(robot, theta, task)):
                # extract the end-effector position from g
                g = robot.fk(theta, visual = True, collision = True)
                end_effector_pose = tuple(round(coord, rounding) for coord in g[:3, 3].flatten())
                valid_poses.add(end_effector_pose)
                
        return list(valid_poses)

    def find_reachibility_percentage(valid_poses, world_dim = [1.00, 1.00, 1.00], world_res = 0.01):
        """Calculate the percentage of the world space that is reachable by our robot configuration.

        Args:
            valid_poses (list[tuple[float, float, float]]): list of (x,y,z) poses that the end effector can reach.
                We assume that the poses are rounded to the world_resolution!
            world_dim (list[float, float, float]): _description_
            world_res (float): how much to split the world (eg, 0.01m increments)

        Returns:
            float: percentage of world space that is reachable
        """
        total_cubes = (world_dim[0] / world_res) * (world_dim[1] / world_res) * (world_dim[2] / world_res)
        print("total cubes", total_cubes)
        reachable_count = len(valid_poses)
        print("reachable count: ", reachable_count)
        return round((reachable_count / total_cubes) * 100, 2)
    
    robot = assembly.to_pin_robot()
    #perform reachability calculations
    valid_poses = find_all_valid_poses(robot, angle_interval = 250)

    return find_reachability_percentage(valid_poses)

def fitness_function(assembly: ModuleAssembly, ga_instance: pygad.GA, index: int) -> Lexicographic:
    """
    This fitness function returns a lexicographic value, where the
    
    - first value indicates the number of modules in the robot, and
    - the second value the minus of the total mass
    """
    #TODO - include reachability metric and max weight calculations in this fitness function

    if assembly.nJoints != 3:
        return -10000
    #this function is an example function.
    return Lexicographic(reachability_metric(assembly), len(assembly.module_instances), -assembly.mass)



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