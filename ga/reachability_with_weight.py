from multiprocessing import Pool
import numpy as np
import timor
from timor.Module import *
from timor.utilities.visualization import animation
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from pathlib import Path
import pickle
import plotly.graph_objects as go # interactive graphs


import threading
from timor.utilities import prebuilt_robots
from math import ceil

class Reachability_with_weight():
    def __init__(self, robot, task=None, angle_interval=100, world_resolution=0.01):
        """Construct a reachability object

        Args:
            robot (timor.Robot.PinRobot): our robot
            task (timor.task.Task, optional): Task containing obstacles. Defaults to None.
            angle_interval (int, optional): how many times do we split the range of angles from 0 to 2pi. Defaults to 100.
            world_resolution (float, optional): used to round the valid poses. Defaults to 0.01.
        """
        self.robot = robot
        self.task = task
        self.angle_interval = angle_interval
        self.world_res = world_resolution
        self.rounding = int(-math.log10(world_resolution))
        
        self.valid_poses = []
        
        
    def check_pose_and_get_gripper_with_weight(self, theta, mass):
        """Evaluate whether it is a valid pose (contains no collision) and if the robot can hold up the task-specified weight.

        Args:
            theta (list[float x n]): joint angles configuration
            mass (int): mass (kg) of the object to be picked up. Assumes gripper will be able to pick up.
                - TODO: find a way to get this from the timor.task if possible, but really not necessary unless gripper physics becomes important

        Returns:
            tuple[float, float, float]: the pose of the end effector if valid
            None: otherwise 
            
        Note:
            TODO: need to include RRT check
        """
        robot = self.robot
        g = self.robot.fk(theta, collision=True)
        self_collision = self.robot.has_self_collision()
        collisions = False if self.task is None else self.robot.has_collisions(self.task, safety_margin=0) # TODO may need to alter safety margin
        valid_pose = not (collisions or self_collision)


        # adding a downward end effector force
        weight = mass * 9.8 # assume on Earth mass --> weight
        eef_wrench = np.array([-0, -0, -weight, 0, 0, 0])  # A downward force of weight Newtons in the Z direction

        # test weight on current robot config
        # Get the joint torque limits from the robot
        joint_torque_limits = robot.joint_torque_limits  # This could be either 1D or 2D

        # Check if it's 1D or 2D
        if joint_torque_limits.ndim == 1:
            # If it's 1D, assume the lower limits are the negatives of the upper limits
            torque_limits = np.array([ -joint_torque_limits, joint_torque_limits]).T
        elif joint_torque_limits.ndim == 2:
            # If it's 2D, assume it's already in the format [lower_limits, upper_limits]
            torque_limits = joint_torque_limits.T

        proposed_joint_torques = robot.id(q=None, dq=None, ddq=None, motor_inertia=True, friction=True, eef_wrench=eef_wrench)
        weight_feasible = self.check_weight_feasibility_with_limits(computed_torques=proposed_joint_torques, torque_limits=torque_limits )
        
        

        
        if valid_pose and weight_feasible:
            end_effector_pose = tuple(round(coord, self.rounding) for coord in g[:3, 3].flatten())
            return end_effector_pose
        else:
            return None
        
    
    def find_all_valid_poses(self):
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
        num_joints = len(self.robot.joints)
        angles = np.linspace(0, 2 * np.pi, self.angle_interval)
        all_angles_combo = itertools.product(angles, repeat=num_joints)
        
        print(f"Angles = {angles}")
        
        # To store all valid poses, multiple joint combos may result in the same end-effector position
        valid_poses = []
            
        for theta in tqdm(list(all_angles_combo), desc="Finding Reachable Workspace"):
            end_effector_pose = self.check_pose_and_get_gripper(theta)
            if (end_effector_pose is not None):
                valid_poses.append(end_effector_pose)
        
        self.valid_poses = valid_poses
        return list(valid_poses)
        
        
    def find_all_valid_poses_multiprocessor_pool(self):
        """Finds all the valid poses (end effector position) that are reachable by 
           iterating through the joint space

        Returns:
            list[tuple[float, float, float]]: a list containing all valid poses (x, y, z) in the world space
        
        Note: 
            this does not work because you don't different processes do not have access to shared variables
        """
        # Generate all possible combinations of joint angles
        num_joints = len(self.robot.joints)
        angles = np.linspace(0, 2 * np.pi, self.angle_interval)
        all_angles_combo = product(angles, repeat=num_joints)
        
        valid_poses = []
        
        with Pool() as p: 
            valid_poses = p.map(self.check_pose_and_get_gripper, all_angles_combo)
            
        return list(valid_poses)
        
    
    def find_all_valid_poses_multithreading(self, num_threads):
        """Finds all the valid poses (end effector position) that are reachable by
           iterating through the joint space (FK)
           
        Returns:
            list[tuple[float, float, float]]: a list containing all valid poses (x, y, z) in the world space
            
        if this doesn't work istg
        """
        print("hi")
        num_joints = len(self.robot.joints)
        angles = np.linspace(0, 2 * np.pi, self.angle_interval)
        all_angles_combo = list(product(angles, repeat=num_joints))
        thetas_work = np.array_split(all_angles_combo, num_threads)
        
        threads = []
        all_valid_poses = []
        
        print("hello")
        
        def simplified_fk(robot, thetas):
            T = np.eye(4)
            for joint, theta in zip(robot.joints, thetas):
                R = joint.get_rotation_matrix(theta)
                p = joint.translation
                T_joint = np.block([[R, p.reshape(3, 1)],
                                    [np.zeros(3), 1]])
                T = T @ T_joint
            return T[:3, 3]  # Return only the position vector
        
        def thread_function(thetas_group):
            valid_poses = []
            
            for t in tqdm(thetas_group):
                g = simplified_fk(self.robot.joints, t)
                # self_collision = self.robot.has_self_collision()
                self_collision = False
                collisions = False if self.task is None else self.robot.has_collisions(self.task, safety_margin=0) # TODO may need to alter safety margin
                valid_pose = not (collisions or self_collision)
        
                if valid_pose:
                    end_effector_pose = tuple(round(coord, self.rounding) for coord in g[:3, 3].flatten())
                    valid_poses.append(end_effector_pose)
            
            # Only one thread writes to the all_valid_poses at a time
            with lock:
                all_valid_poses.extend(valid_poses)
                
        lock = threading.Lock()
        
        print("hola")
        
        for t in range(num_threads):
            print(f"{t} thread created")
            thread = threading.Thread(target=thread_function, args=(thetas_work[t],))
            threads.append(thread)
            thread.start()
            
        print("Waiting for threads")
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        self.valid_poses = all_valid_poses
        return all_valid_poses
        
        
    def plot_reachability(self, filename="reachability_plot.png"):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Convert lists to numpy arrays for easier plotting
        reachable_points = np.array(self.valid_poses)

        # Plot reachable points in green
        if len(reachable_points) > 0:
            ax.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2], color='green', label='Reachable', s=5)

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
        # Convert lists to numpy arrays for easier plotting
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
            title='3D Reachability Plot'
        )

        fig.show()
        
    def reachability_random_sample(self, num_samples, mass=1):
        """Reachable points with their manipulability index."""
        reachable_points = set()

        for _ in tqdm(range(num_samples), disable=True):
            q = self.robot.random_configuration()
            end_effector_pose = self.check_pose_and_get_gripper_with_weight(q, mass)
            if not self.robot.has_self_collision(q) and not end_effector_pose:
                #tcp_pose = self.robot.fk(q)
                #end_effector_pose = tuple(round(coord, 2) for coord in tcp_pose[:3, 3].flatten())
                
                # Compute the Yoshikawa manipulability index
                # High Yoshikawa manipulability index values indicate that the robot has good dexterity 
                # and can move in multiple directions from the corresponding configurations. 
                # Low manipulability index values indicate that corresponding configurations are less 
                # dexterous and more susceptible to singularities.
                manipulability_idk = self.robot.manipulability_index(q)
                reachable_points.add((end_effector_pose, manipulability_idk))  
                
        return list(reachable_points)
    
    
    def plot_interactive_reachability_with_manipulability(self, reachable, manipulability):
        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reachable[:, 0],
            y=reachable[:, 1],
            z=reachable[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=manipulability,
                colorscale='viridis',
                opacity=0.8,
                colorbar=dict(title='Manipulability Index')
            ),
            hovertext=[f'Manipulability: {m:.3f}' for m in manipulability],
            hoverinfo='text'
        )])

        # Update the layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='Robot Reachability Space with Manipulability Index'
        )

        # Show the interactive plot
        fig.show()
        

    def find_reachibility_percentage(self, voxel_size = 0.1, valid_pose = None):
        """Calculate the percentage of the world space that is reachable by our robot configuration.

        Args:
            valid_poses ndarray of (3, ): (x,y,z) poses that the end effector can reach
            voxel_size (float): used to split the space

        Returns:
            float: percentage of world space that is reachable (occupied voxels / total voxels)
        """
        voxel_x = ceil(self.world_dim[0] / voxel_size)
        voxel_y = ceil(self.world_dim[1] / voxel_size)
        voxel_z = ceil(self.world_dim[2] / voxel_size)
        voxel_grid = np.zeros((voxel_x, voxel_y, voxel_z), dtype = bool)
            
        for x, y, z in valid_pose: 
            vx = int( (x / voxel_size) + (voxel_x / 2) )
            vy = int( (y / voxel_size) + (voxel_y / 2) )
            vz = int( (z / voxel_size) + (voxel_z / 2) )
            
            if (0 <= vx < voxel_x) and (0 <= vy < voxel_y) and (0 <= vz < voxel_z):
                voxel_grid[vx, vy, vz] = True
                        
        occupied_voxels = np.sum(voxel_grid)
        total_voxels = voxel_x * voxel_y * voxel_z     
           
        # print("reachable count: ", occupied_voxels)
        # print("num voxels: ", total_voxels)
        
        # # --- plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.voxels(voxel_grid, edgecolor='k')

        # ax.set_xlabel('X-axis')
        # ax.set_ylabel('Y-axis')
        # ax.set_zlabel('Z-axis')

        # plt.title('3D Voxel Grid')
        # plt.show()
        
        
        return round(occupied_voxels / total_voxels * 100, 2)        
    # def find_reachibility_percentage(self, world_dim = [1.00, 1.00, 1.00], world_res = 0.01):
    #     """Calculate the percentage of the world space that is reachable by our robot configuration.

    #     Args:
    #         valid_poses (list[tuple[float, float, float]]): list of (x,y,z) poses that the end effector can reach.
    #             We assume that the poses are rounded to the world_resolution!
    #         world_dim (list[float, float, float]): _description_
    #         world_res (float): how much to split the world (eg, 0.01m increments)

    #     Returns:
    #         float: percentage of world space that is reachable
    #     """
    #     total_cubes = (world_dim[0] / world_res) * (world_dim[1] / world_res) * (world_dim[2] / world_res)
    #     print("total cubes", total_cubes)
    #     reachable_count = len(self.valid_poses)
    #     print("reachable count: ", reachable_count)
    #     return round((reachable_count / total_cubes) * 100, 2)
    


    def check_weight_feasibility_with_limits(computed_torques: np.ndarray, torque_limits: np.ndarray) -> bool:
        """
        Checks if the computed torques are feasible given the upper and lower allowable torque limits for each joint.

        Parameters:
            computed_torques (numpy.ndarray): The torques computed by the static_torque function (n_joints x 1).
            torque_limits (numpy.ndarray): A 2D array containing the lower and upper limits for each joint's torque.
                                        Shape: (n_joints, 2) where each row contains [lower_limit, upper_limit].

        Returns:
            bool: True if all computed torques are within the allowable limits, False otherwise.
        """
        # Check if the computed torques are within the lower and upper limits for each joint
        lower_limits = torque_limits[:, 0]
        upper_limits = torque_limits[:, 1]


        if torque_limits:
            # Check if all computed torques are within their respective limits
            feasible = np.all((computed_torques >= lower_limits) & (computed_torques <= upper_limits))
        else:
            feasible = False

        
        ## Debugging prints
        # if feasible:
        #     print("All computed torques are within the allowable limits.")
        # else:
        #     # Identify the joints where the torque exceeds the limits
        #     exceeding_joints_lower = np.where(computed_torques < lower_limits)[0]
        #     exceeding_joints_upper = np.where(computed_torques > upper_limits)[0]
            
            # Print which joints exceed the torque limits
            # if len(exceeding_joints_lower) > 0:
            #     print(f"Joints {exceeding_joints_lower} have computed torques below the lower limit.")
            # if len(exceeding_joints_upper) > 0:
            #     print(f"Joints {exceeding_joints_upper} have computed torques above the upper limit.")
        
        return feasible
            

    