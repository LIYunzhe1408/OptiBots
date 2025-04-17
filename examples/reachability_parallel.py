import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from pathlib import Path
import plotly.graph_objects as go 
from math import ceil
import ray
from scipy.spatial import KDTree

import timor
from timor.utilities import prebuilt_robots
from timor.task.Task import *
from timor.task.Obstacle import *
from timor.Module import *
from timor.utilities.visualization import animation
from timor.Geometry import Box, ComposedGeometry, Cylinder


# constants
SAFETY_MARGIN = 0.2
EE_DOWNWARDS_AXIS = [0, 0, -1]

@ray.remote
class ReachabilityWorker:
    def __init__(self, robot_create, task_create, mass):
        """Initialize a worker with its own robot and task instances."""
        self.robot = robot_create()
        self.task = task
        self.mass = mass
        self.safety_margin = SAFETY_MARGIN
        self.ee_downwards_axis = EE_DOWNWARDS_AXIS
        self.valid_configs = set()
        self.valid_positions = []
    
    
    def process_batch(self, batch_size):
        """Process a batch of random samples and return valid positions."""
        valid_positions = set()
        valid_configs = set()
        
        for _ in range(batch_size):
            # Generate random configuration
            q = tuple(round(joint, 3) for joint in self.robot.random_configuration())
            
            # Skip if already checked
            if q in valid_configs:
                continue
            
            # Forward kinematics to get TCP pose
            tcp_pose = self.robot.fk(q)
            
            # 1. Check end effector orientation
            end_effector_z_axis = tcp_pose[:3, 2]
            if all(end_effector_z_axis == self.ee_downwards_axis):
                continue
            
            # 2. Check self-collision
            if self.robot.has_self_collision(q):
                continue
                
            # 3. Check obstacle collision
            if self.task and self.robot.has_collisions(self.task, safety_margin=self.safety_margin):
                continue
            
            # 4. Check weight feasibility
            weight = self.mass * 9.8
            eef_wrench = np.array([0, 0, -weight, 0, 0, 0])
            joint_torque_limits = self.robot.joint_torque_limits
            if joint_torque_limits.ndim == 1:
                torque_limits = np.array([-joint_torque_limits, joint_torque_limits]).T
            elif joint_torque_limits.ndim == 2:
                torque_limits = joint_torque_limits.T
            proposed_joint_torques = self.robot.id(q=None, dq=None, ddq=None, 
                                                 motor_inertia=True, friction=True, 
                                                 eef_wrench=eef_wrench)
            weight_feasible = np.all((proposed_joint_torques >= torque_limits[:, 0]) & 
                                    (proposed_joint_torques <= torque_limits[:, 1]))
            if not weight_feasible:
                continue
            
            # 5. Add valid pposition of the end effector
            ee_position = tuple(round(p, 3) for p in tcp_pose[:3, 3].flatten())
            valid_positions.add(ee_position)
            valid_configs.add(q)
            
        self.valid_configs = list(valid_configs)
        self.valid_positions = list(valid_positions)
        
        return valid_positions
    

class Reachability:
    def __init__(self, robot_module, task = None, mass = 1, angle_interval = 20, world_min_dim = [-0.5, -0.5, 0.0], world_max_dim = [0.5, 0.5, 1.0]):
        """Construct a reachability object

        Args:
            robot (timor.Robot.PinRobot): our robot
            task (timor.task.Task, optional): Task containing obstacles. Defaults to None.
            mass (int): mass (kg) of the object to be picked up. Assumes gripper will be able to pick up.
                - TODO: find a way to get this from the timor.task if possible, but really not necessary unless gripper physics becomes important
            angle_interval (int, optional): how many times do we split the range of angles from 0 to 2pi. Defaults to 100.
            world_dimension (list of floats): dimmension of the world arround robot.
            world_resolution (float, optional): used to round the valid poses. Defaults to 0.01.
        """
        self.robot_module = robot_module
        self.task = task
        self.angle_interval = angle_interval
        self.world_min_dim = world_min_dim
        self.world_max_dim = world_max_dim
        
        self.ee_downwards_axis = [0,0,-1]
        self.mass = mass
        self.safety_margin = 0.2
        self.obstacles_found = 0
        
        self.valid_configs = set()
        self.valid_positions = []
        
    def reachability_random_sample_actors(self, num_samples, num_actors = 4, batch_size = 1000):
        """Parallel implementation of reachable points sampling using Ray actors."""
        if not ray.is_initialized():
            ray.init()
            
        # Create robot and task factory functions
        def robot_factory():
            # Create a new instance of the robot
            # This will depend on how your robot is initialized
            # return prebuilt_robots.get_six_axis_modrob()
            return self.robot_module.to_pin_robot()
        
        # Create actors
        actors = []
        for _ in range(num_actors):
            actors.append( ReachabilityWorker.remote(robot_create = robot_factory, task_create = self.task, mass = self.mass))
        batches_per_actor = ceil(num_samples / (num_actors * batch_size))
        total_batches = num_actors * batches_per_actor
        
        # Launch parallel processing
        print(f"Launching {total_batches} batches across {num_actors} actors...")
        futures = []
        for actor in actors:
            for _ in range(batches_per_actor):
                futures.append(actor.process_batch.remote(batch_size))
        
        # Collect results with progress tracking
        all_valid_positions = []
        
        for i, future_result in enumerate(ray.get(futures)):
            valid_positions = future_result
            all_valid_positions.extend(valid_positions)
            
            if (i+1) % num_actors == 0:
                batch_num = (i+1) // num_actors
                print(f"Batch {batch_num}/{batches_per_actor} complete. Found {len(valid_positions)} valid poses.")
        
        # Update the reachability object with results, avoiding repetition
        self.valid_positions = np.array(list(set(all_valid_positions)))
        print(f"Total valid poses found: {len(self.valid_positions)}")
        return self.valid_positions
    
    
    def plot_reachability(self, filename = "reachability_plot.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        reachable_points = np.array(self.valid_positions)
    
        # Plot reachable points in green
        if len(reachable_points) > 0:
            ax.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2], color='green', label='Reachable', s=5)
        else:
            print("there are no reachable points")

        # Set plot labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Reachability Plot')
        ax.legend()

        # Save the plot to a file
        plt.savefig(filename)  
        plt.close(fig) 
        print(f"Reachability plot saved to {filename}")
        
        
    def find_reachibility_percentage(self, voxel_size = 0.1):
        """Calculate the percentage of the world space that is reachable by our robot configuration.

        Args:
            voxel_size (float): used to split the space

        Returns:
            float: percentage of world space that is reachable (occupied voxels / total voxels)
        """
        # calculate num of bins in each dimension
        num_voxels_x = ceil((self.world_max_dim[0] - self.world_min_dim[0])/ voxel_size)
        num_voxels_y = ceil((self.world_max_dim[1] - self.world_min_dim[1])/ voxel_size)
        num_voxels_z = ceil((self.world_max_dim[2] - self.world_min_dim[2])/ voxel_size)
        voxel_grid = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z), dtype = bool)
                    
        for x, y, z in self.valid_positions: 
            # find idx in voxel grid. Shift coordinate and then calculate voxel
            vx = int((x - self.world_min_dim[0]) / voxel_size)
            vy = int((y - self.world_min_dim[1]) / voxel_size)
            vz = int((z - self.world_min_dim[2]) / voxel_size)
            
            if (0 <= vx < num_voxels_x) and (0 <= vy < num_voxels_y) and (0 <= vz < num_voxels_z):
                voxel_grid[vx, vy, vz] = True
                        
        occupied_voxels = np.sum(voxel_grid)
        total_voxels = num_voxels_x * num_voxels_y * num_voxels_z     
           
        print("reachable count: ", occupied_voxels)
        print("num voxels: ", total_voxels)
        
        # --- plot for debugging only ---
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.voxels(voxel_grid, color='red', edgecolor='k')

        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')

        # plt.title('3D Voxel Grid')
        # plt.show()
        
        return round(occupied_voxels / total_voxels * 100, 2)
    
    def specific_pose_valid(self, theta):
        """
        Given a robot at a specific pose, evaluate whether it is a valid pose (contains no collision).
        TODO: reach target pose from the previous pose

        Args:
            robot (timor.Robot.PinRobot): robot
            theta (list[float]): joint angles configuration

        Returns:
            The pose of the end effector if valid, otherwise returns None
        """	
        if theta in self.valid_configs:
            return True
        
        robot = self.robot
        g = self.robot.fk(theta, collision=True)
        self_collision = robot.has_self_collision()
        collisions = False if self.task is None else robot.has_collisions(self.task, safety_margin = 0) # TODO may need to alter safety margin
        valid_pose = not (collisions or self_collision)
        
        if valid_pose:
            self.valid_configs.add(theta)
        
        return valid_pose
    

    def find_trajectory_bidirectional_kdtree(self, base_config, target_config, max_rrt_iters=5000, rrt_step_size=0.1, target_distance_thresh=0.3):
        forward_tree = [base_config]
        backward_tree = [target_config]
        
        # Track parents for path reconstruction
        forward_parents = {0: -1}
        backward_parents = {0: -1}
        
        joint_limits = self.robot.joint_limits
        low_bounds, high_bounds = joint_limits[0], joint_limits[1]
        
        for i in range(max_rrt_iters):
            forward_kdtree = KDTree(forward_tree)
            backward_kdtree = KDTree(backward_tree)
            
            # Grow forward tree
            random_config = np.random.uniform(low_bounds, high_bounds)
            forward_idx = forward_kdtree.query(random_config.reshape(1, -1), k=1)[1][0]
            nearest_config = forward_tree[forward_idx]
            
            # Step toward random configuration
            direction = random_config - nearest_config
            norm = np.linalg.norm(direction)
            if norm > 0:
                new_config = np.array([round(j, 3) for j in (nearest_config + rrt_step_size * direction / norm)])
                
                if self.specific_pose_valid(tuple(new_config)):
                    forward_parents[len(forward_tree)] = forward_idx
                    forward_tree.append(new_config)
                    
                    # Try to connect backward tree to this new node
                    backward_idx = backward_kdtree.query(new_config.reshape(1, -1), k=1)[1][0]
                    backward_node = backward_tree[backward_idx]
                    
                    if np.linalg.norm(new_config - backward_node) < target_distance_thresh:
                        # Trees are close enough - no need to reconstruct path
                        return True
            
            # Swap trees and repeat (alternating growth)
            forward_tree, backward_tree = backward_tree, forward_tree
            forward_parents, backward_parents = backward_parents, forward_parents
            forward_kdtree, backward_kdtree = backward_kdtree, forward_kdtree
        
        return None

    
    
   
        
if __name__ == "__main__":
    ray.init()
    
    # Create robot and task as before
    # default_robot = prebuilt_robots.get_six_axis_modrob()
    
    ## 3 AXIS ROBOT
    # Create a database
    from timor.utilities.file_locations import get_module_db_files
    modules_file = get_module_db_files('geometric_primitive_modules')
    db = ModulesDB.from_json_file(modules_file) 
    modules = ('base', 'i_30', 'J2', 'J2', 'J2', 'i_30', 'eef')
    B = ModuleAssembly.from_serial_modules(db, modules)
    # long_robot = B.to_pin_robot() #convert to pinocchio robot
    
    # Create task
    task_header = TaskHeader(
        ID='Dummy Cube',
        tags=['test'],
        author=['AZJ']
    )
    
    cube = Obstacle(ID = 'cube', collision = Box(
        parameters = dict(x = 0.2, y = 0.2, z = 0.2), 
        pose = Transformation.from_translation([0.5, 0.5, 0.5])
    ))
    task = Task(task_header, obstacles = [cube])
    
    # Create reachability object
    reachability = Reachability(
        robot_module = B, 
        task = task,
        world_min_dim = [-1.5, -1.5, -1.5], 
        world_max_dim = [2.0, 2.0, 2.0],
    )
    
    # Run parallel sampling with actors
    start_t = time.time()
    reachability.reachability_random_sample_actors(num_samples = 100000, num_actors = 4, batch_size = 1000)
    print(f"Time to find reachability: {time.time() - start_t} seconds")
    
    # Visualize and analyze results
    reachability.plot_reachability("reachability_plot_parallel.png")
    percentage = reachability.find_reachibility_percentage(voxel_size = 0.1)
    print(f"Percentage of reachability: {percentage}%")
    
    ray.shutdown()

    