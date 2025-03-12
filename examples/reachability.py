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
from math import ceil


import threading
from timor.utilities import prebuilt_robots


class Reachability():    
    def __init__(self, robot, task = None, angle_interval = 20, world_dimension = [1.0, 1.0, 1.0], world_resolution = 0.01):
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
        self.world_dim = world_dimension
        self.world_res = world_resolution
        self.rounding = int(-math.log10(world_resolution))
        
        self.ee_downwards_axis = [0,0,-1]
        self.safety_margin = 0.2
        
        self.valid_poses = []
        
        
    def check_pose_and_get_gripper(self, theta):
        """Evaluate whether it is a valid pose (contains no collision).

        Args:
            theta (list[float x n]): joint angles configuration

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
        
        if valid_pose:
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
        
        
    def plot_reachability(self, reachable = None, filename="reachability_plot.png"):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Convert lists to numpy arrays for easier plotting
        if (type(reachable) == None):
            reachable_points = np.array(self.valid_poses)
        else:
            reachable_points = np.array(reachable)
        
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
        
    def plot_interactive_reachability(self, reachable = None):
        # Convert lists to numpy arrays for easier plotting
                # Convert lists to numpy arrays for easier plotting
        if (type(reachable) == None):
            reachable_points = np.array(self.valid_poses)
        else:
            reachable_points = np.array(reachable)

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
        
    def reachability_random_sample(self, num_samples, task = None):
        """Reachable points with their manipulability index."""
        reachable_points = set()

        for _ in tqdm(range(num_samples)):
            q = self.robot.random_configuration()
            tcp_pose = self.robot.fk(q)
            
            # Check if end effector is fully pointing downwards.
            # The 3rd column of the rotation matrix corresponds to the z-axis of the EE w.r.t. world frame
            end_effector_z_axis = tcp_pose[:3, 2]
            if all(end_effector_z_axis == self.ee_downwards_axis):
                continue
            
            # Check onfig has no collision
            if self.robot.has_self_collision(q):
                continue
                
            # check collision between obstacles and current config (changed in has_self_collision call) 
            if task and self.robot.has_collisions(task, safety_margin = self.safety_margin):
                continue
                
            end_effector_pose = tuple(round(coord, 2) for coord in tcp_pose[:3, 3].flatten())

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
           
        print("reachable count: ", occupied_voxels)
        print("num voxels: ", total_voxels)
        
        # --- plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(voxel_grid, edgecolor='k')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        plt.title('3D Voxel Grid')
        plt.show()
        
        
        return round(occupied_voxels / total_voxels * 100, 2)
       
        
        
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
    world_dimension = [3.20, 3.20, 3.20]
    default_robot = prebuilt_robots.get_six_axis_modrob()
    reachability = Reachability(robot = default_robot, 
                                angle_interval = how_many_times_to_split_angle_range, 
                                world_resolution = world_resolution, 
                                world_dimension = world_dimension)
    
    start_t = time.time()
    valid_poses = reachability.reachability_random_sample(num_samples = 100000)
    # valid_poses = reachability.find_all_valid_poses_multithreading(num_threads)
    print(f"Time to find reachability: {time.time() - start_t} seconds")
    
    reachable, manipulability =  zip(*valid_poses)
    reachable = np.array([list(pt) for pt in reachable])
    print("reachable raw ", reachable)
    
    reachability.plot_interactive_reachability_with_manipulability(reachable, manipulability)
    percentage = reachability.find_reachibility_percentage(voxel_size = 0.5, valid_pose = reachable)
    
    print(f"percentage {percentage}%")
    
    # percentage = reachability.find_reachibility_percentage()
    # print(f"Percentage of reachability: {percentage}%")
    