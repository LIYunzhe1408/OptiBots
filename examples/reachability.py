import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from pathlib import Path
import plotly.graph_objects as go # interactive graphs
from math import ceil
from scipy.spatial import KDTree

import timor
from timor.utilities import prebuilt_robots
from timor.task.Task import *
from timor.task.Obstacle import *
from timor.Module import *
from timor.utilities.visualization import animation
from timor.Geometry import Box, ComposedGeometry, Cylinder




class Reachability():    
    def __init__(self, robot, mass = 1, task = None, angle_interval = 20, world_min_dim = [-0.5, -0.5, 0.0], world_max_dim = [0.5, 0.5, 1.0], include_rrt = False):
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
        self.robot = robot
        self.task = task
        self.angle_interval = angle_interval
        self.world_min_dim = world_min_dim
        self.world_max_dim = world_max_dim
        # self.world_res = world_resolution
        # self.rounding = int(-math.log10(world_resolution))
        
        self.include_rrt = include_rrt
        self.ee_downwards_axis = [0,0,-1]
        self.mass = mass
        self.safety_margin = 0.2
        self.obstacles_found = 0
        
        self.valid_configs = set()
        self.valid_poses = []
        
    
    def reachability_random_sample(self, num_samples):
        """Reachable points with their manipulability index."""
        initial_config = self.robot.configuration
        reachable_points = set()
        reachable_configs = set()

        for _ in tqdm(range(num_samples)):            
            q = tuple(round(joint, 3) for joint in self.robot.random_configuration())
            tcp_pose = self.robot.fk(q)
            
            if q in reachable_configs: 
                print("already checked")
                continue
            
            # 1. Check end effector is fully pointing downwards.
            # The 3rd column of the rotation matrix corresponds to the z-axis of the EE w.r.t. world frame
            end_effector_z_axis = tcp_pose[:3, 2]
            if all(end_effector_z_axis == self.ee_downwards_axis):
                continue
            
            # 2. Check config has no self collision
            if self.robot.has_self_collision(q):
                continue
                
            # 3. Check collision between obstacles and current config
            if self.task and self.robot.has_collisions(task, safety_margin = self.safety_margin):
                continue
            
            # 4. Check that a given mass can be lifted
            # adding a downward end effector force
            weight = self.mass * 9.8 # assume on Earth mass --> weight
            eef_wrench = np.array([-0, -0, -weight, 0, 0, 0])  # A downward force of weight Newtons in the Z direction

            # test weight on current robot config
            # Get the joint torque limits from the robot
            joint_torque_limits = self.robot.joint_torque_limits  # This could be either 1D or 2D

            # Check if it's 1D or 2D
            if joint_torque_limits.ndim == 1:
                # If it's 1D, assume the lower limits are the negatives of the upper limits
                torque_limits = np.array([ -joint_torque_limits, joint_torque_limits]).T
            elif joint_torque_limits.ndim == 2:
                # If it's 2D, assume it's already in the format [lower_limits, upper_limits]
                torque_limits = joint_torque_limits.T

            proposed_joint_torques = self.robot.id(q = None, dq = None, ddq = None, motor_inertia = True, friction = True, eef_wrench = eef_wrench)
            weight_feasible = np.all((proposed_joint_torques >= torque_limits[:, 0]) & (proposed_joint_torques <= torque_limits[:, 1]))
            if not weight_feasible:
                continue
            
            # 5. Check if there exists a trajectory from initial configuration to new config pose
            # end_effector_pose = tuple(round(coord, 2) for coord in tcp_pose[:3, 3].flatten())
            if self.include_rrt:
                path_found = self.find_trajectory_bidirectional_kdtree(base_config = initial_config, target_config = q)
                if not path_found:
                    continue
                else:
                    print('rrt path found')

            # Compute the Yoshikawa manipulability index
            # High Yoshikawa manipulability index values indicate that the robot has good dexterity 
            # and can move in multiple directions from the corresponding configurations. 
            # Low manipulability index values indicate that corresponding configurations are less 
            # dexterous and more susceptible to singularities.
            # manipulability_idk = self.robot.manipulability_index(q)
            # reachable_points.add((end_effector_pose, manipulability_idk)) 
            
            # 6. DONE. Get the EE pose 
            ee_pose = tuple(tcp_pose[:3, 3].flatten())
            reachable_points.add(ee_pose) 
            self.valid_configs.add(tuple(q))
            
            
        # self.reachable, self.manipulability =  zip([reachable_points])
        self.valid_poses = np.array(list(reachable_points))
        # print(self.valid_poses)
        return self.valid_poses
        
        
    # def check_weight_feasibility_with_limits(self, computed_torques: np.ndarray, torque_limits: np.ndarray) -> bool:
    #     """
    #     Checks if the computed torques are feasible given the upper and lower allowable torque limits for each joint.

    #     Parameters:
    #         computed_torques (np.ndarray): The torques computed by the static_torque function (n_joints x 1).
    #         torque_limits (np.ndarray): A 2D array containing the lower and upper limits for each joint's torque.
    #                                     Shape: (n_joints, 2) where each row contains [lower_limit, upper_limit].

    #     Returns:
    #         bool: True if all computed torques are within the allowable limits, False otherwise.
    #     """
    #     # Check if the computed torques are within the lower and upper limits for each joint
    #     lower_limits = torque_limits[:, 0]
    #     upper_limits = torque_limits[:, 1]

    #     if torque_limits is not None and torque_limits.size > 0:
    #         # Check if all computed torques are within their respective limits
    #         feasible = np.all((computed_torques >= lower_limits) & (computed_torques <= upper_limits))
    #     else:
    #         feasible = False

    #     return feasible
        
    # def check_pose_and_get_gripper_with_weight(self, theta, mass):
    #     """Evaluate whether it is a valid pose (contains no collision) and if the robot can hold up the task-specified weight.

    #     Args:
    #         theta (list[float x n]): joint angles configuration
    #         mass (int): mass (kg) of the object to be picked up. Assumes gripper will be able to pick up.
    #             - TODO: find a way to get this from the timor.task if possible, but really not necessary unless gripper physics becomes important

    #     Returns:
    #         tuple[float, float, float]: the pose of the end effector if valid
    #         None: otherwise 
            
    #     Note:
    #         TODO: need to include RRT check
    #     """
    #     robot = self.robot
    #     g = self.robot.fk(theta, collision=True)
    #     self_collision = self.robot.has_self_collision()
    #     collisions = False if self.task is None else self.robot.has_collisions(self.task, safety_margin=0) # TODO may need to alter safety margin
    #     valid_pose = not (collisions or self_collision)


    #     # adding a downward end effector force
    #     weight = mass * 9.8 # assume on Earth mass --> weight
    #     eef_wrench = np.array([-0, -0, -weight, 0, 0, 0])  # A downward force of weight Newtons in the Z direction

    #     # test weight on current robot config
    #     # Get the joint torque limits from the robot
    #     joint_torque_limits = robot.joint_torque_limits  # This could be either 1D or 2D

    #     # Check if it's 1D or 2D
    #     if joint_torque_limits.ndim == 1:
    #         # If it's 1D, assume the lower limits are the negatives of the upper limits
    #         torque_limits = np.array([ -joint_torque_limits, joint_torque_limits]).T
    #     elif joint_torque_limits.ndim == 2:
    #         # If it's 2D, assume it's already in the format [lower_limits, upper_limits]
    #         torque_limits = joint_torque_limits.T

    #     proposed_joint_torques = robot.id(q = None, dq = None, ddq = None, motor_inertia = True, friction = True, eef_wrench = eef_wrench)
    #     weight_feasible = self.check_weight_feasibility_with_limits(computed_torques = proposed_joint_torques, torque_limits = torque_limits)
        
        # if valid_pose and weight_feasible:
        #     end_effector_pose = tuple(round(coord, self.rounding) for coord in g[:3, 3].flatten())
        #     return end_effector_pose
        # else:
        #     return None
        
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
        
        
    def find_trajectory(self, base_config, target_config, max_rrt_iters: int = 10000, rrt_step_size: float = 0.1, target_distance_thresh: float = 0.2, goal_bias: float = 0.2):
        '''
        Checks if a target pose is reachable from a base pose through any valid trajectory. 
        Returns a valid trajectory if found. Does not guarantee anything close to optimality.
        
        This implementation uses Rapidly-exploring Random Trees (RRT).

        IMPORTANT - this function works entirely within the joint space. no calculation of actual end effector position is done. this matters for the target distance threshold!

        Parameters:
            - target_config
            - base_config
            - max_rrt_iters
            - rrt_step_size
            - target_distance_thresh

        Returns:
            - the path taken if a trajectory exists. None otherwise

        TODO:
            - make the tree an actual tree - this impl is a little scuffed
        '''

        #if the base or target poses are invalid, no trajectory is possible.	
        # if not specific_pose_valid(robot, base_config, task) or not specific_pose_valid(robot, target_config, task):
        # 	return None

        #each node in the tree contains a tuple of (joint_angles: List[float], parent: Integer) - parent is the parent index 
        tree = [base_config]
        idx_to_parent = dict()
        idx_to_parent[0] = -1

        #nx2 arr - of lower and upper angle limits for each joint
        joint_space_bounds = self.robot.joint_limits
        low_bounds, high_bounds = joint_space_bounds[0], joint_space_bounds[1]

        #RRT Algo
        for _ in range(max_rrt_iters):
            # With probability goal_bias, sample the goal directly
            if np.random.random() < goal_bias:
                random_config = target_config
            else:
                random_config = np.random.uniform(low_bounds, high_bounds)
            
            # Find nearest node in the tree
            nearest_config_idx, nearest_config = min(enumerate(tree), key=lambda x: np.linalg.norm(x[1] - random_config))
            
            # Extend towards random config
            new_config = nearest_config + rrt_step_size * (random_config - nearest_config) / np.linalg.norm(random_config - nearest_config)
            #if the new config that we step into is valid, add it to the tree. we can 
            if self.specific_pose_valid(new_config):
                # print(new_config, len(tree))
                #append a new config and its parent's idx in the tree
                idx_to_parent[len(tree)] = nearest_config_idx
                tree.append(new_config)
                
                # Check if we've reached the target. if so, return
                if np.linalg.norm(new_config - target_config) < target_distance_thresh:
                    full_path = [target_config]

                    next_node, parent_idx = new_config, nearest_config_idx

                    while parent_idx != -1:
                        full_path.insert(0, next_node)
                        next_node, parent_idx = tree[parent_idx], idx_to_parent[parent_idx]

                    return full_path
        
        #no trajectories are valid. return None
        return None
        

        
    def plot_reachability(self, filename = "reachability_plot.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        reachable_points = np.array(self.valid_poses)
    
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
        
        
    def plot_interactive_reachability(self):
        reachable_points = np.array(self.valid_poses)

        # Create a 3D scatter plot with Plotly
        fig = go.Figure()

        # Plot reachable points in green
        if len(reachable_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=reachable_points[:, 0],
                y=reachable_points[:, 1],
                z=reachable_points[:, 2],
                mode='markers',
                marker=dict(size=3, color='green'),
                name='Reachable'
            ))

        # Set plot labels and title
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='3D Interactive Reachability Plot'
        )

        fig.show()
    
    
    # def plot_interactive_reachability_with_manipulability(self, reachable, manipulability):
    #     # Create the 3D scatter plot
    #     fig = go.Figure(data=[go.Scatter3d(
    #         x=reachable[:, 0],
    #         y=reachable[:, 1],
    #         z=reachable[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=5,
    #             color=manipulability,
    #             colorscale='viridis',
    #             opacity=0.8,
    #             colorbar=dict(title='Manipulability Index')
    #         ),
    #         hovertext=[f'Manipulability: {m:.3f}' for m in manipulability],
    #         hoverinfo='text'
    #     )])

    #     # Update the layout
    #     fig.update_layout(
    #         scene=dict(
    #             xaxis_title='X',
    #             yaxis_title='Y',
    #             zaxis_title='Z'
    #         ),
    #         title='Robot Reachability Space with Manipulability Index'
    #     )

    #     # Show the interactive plot
    #     fig.show()
        
        
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
        
        print(num_voxels_x, ' x ', num_voxels_y, ' x ', num_voxels_z)
            
        for x, y, z in self.valid_poses: 
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(voxel_grid, color='red', edgecolor='k')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        plt.title('3D Voxel Grid')
        plt.show()
        
        return round(occupied_voxels / total_voxels * 100, 2)
    
    def print_reachable_pts(self):
        for pt in self.valid_poses:
            print(pt)
       
        
        
if __name__ == "__main__":
    
    # Create a database
    from timor.utilities.file_locations import get_module_db_files
    modules_file = get_module_db_files('geometric_primitive_modules')
    db = ModulesDB.from_json_file(modules_file) 
    
    how_many_times_to_split_angle_range = 30
    world_resolution = 0.01
    # world_dimension = [1.00, 1.00, 1.00]
    # num_threads = 5


    ## 3 AXIS ROBOT
    # modules = ('base', 'i_30', 'J2', 'J2', 'J2', 'i_30', 'eef')
    # B = ModuleAssembly.from_serial_modules(db, modules)
    # long_robot = B.to_pin_robot() #convert to pinocchio robot
    # # viz = long_robot.visualize()
    # reachability = Reachability(robot=long_robot, angle_interval=how_many_times_to_split_angle_range, world_resolution=world_resolution)
    
    # start_t = time.time()
    # valid_poses = reachability.reachability_random_sample(num_samples = 100000)
    # # valid_poses = reachability.find_all_valid_poses_multithreading(num_threads)
    # print(f"Time to find reachability: {time.time() - start_t} seconds")
    
    # # reachability.plot_reachability("B_robot_reachability.png")
    # # percentage = reachability.find_reachibility_percentage()
    # # print(f"Percentage of reachability: {percentage}%") # TODO i think this is a wrong implementation
 
    # reachable, manipulability =  zip(*valid_poses)
    # reachable = np.array([list(pt) for pt in reachable])
    # reachability.plot_interactive_reachability_with_manipulability(reachable, manipulability)
    # percentage = reachability.find_reachibility_percentage(voxel_size = 0.1, valid_pose = reachable)

    
    
    
 

    # 6 AXIS ROBOT
    # world_dimension = [3.20, 3.20, 3.20]
    default_robot = prebuilt_robots.get_six_axis_modrob()
    
    # Create a new task
    task_header = TaskHeader(
        ID='Dummy Cube',
        tags=['test'],
        author=['AZJ']
    )
    
    # Add obstacles (as collision objects)
    cube = Obstacle(ID = 'cube', collision = Box(parameters = dict(x = 0.2, y = 0.2, z = 0.2), pose = Transformation.from_translation([0.5, 0.5, 0.5])))
    task = Task(task_header, obstacles=[cube])
    task.visualize()
    
    reachability = Reachability(robot = default_robot, task = task,
                                world_min_dim = [-1.5, -1.5, -1.5], world_max_dim = [2.0, 2.0, 2.0])
    
    start_t = time.time()
    reachability.reachability_random_sample(num_samples = 100000)
    print(f"Time to find reachability: {time.time() - start_t} seconds")
    
    # reachable, manipulability =  zip(*valid_poses)
    # reachable = np.array([list(pt) for pt in reachable])
    # print("reachable raw ", reachable)
    
    # reachability.plot_interactive_reachability_with_manipulability(reachable, manipulability)
    reachability.plot_reachability()
    percentage = reachability.find_reachibility_percentage(voxel_size = 0.1)
    print(f"percentage {percentage}%")
    
    # percentage = reachability.find_reachibility_percentage()
    # print(f"Percentage of reachability: {percentage}%")
    