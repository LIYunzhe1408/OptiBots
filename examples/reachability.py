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

import threading

class Reachability():
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
        num_joints = len(self.robot.joints)
        angles = np.linspace(0, 2 * np.pi, self.angle_interval)
        all_angles_combo = list(product(angles, repeat=num_joints))
        thetas_work = np.array_split(all_angles_combo, num_threads)
        
        threads = []
        all_valid_poses = []
        
        def thread_function(thetas_group):
            valid_poses = []
            
            for t in thetas_group:
                g = self.robot.fk(t, collision=True)
                self_collision = self.robot.has_self_collision()
                collisions = False if self.task is None else self.robot.has_collisions(self.task, safety_margin=0) # TODO may need to alter safety margin
                valid_pose = not (collisions or self_collision)
        
                if valid_pose:
                    end_effector_pose = tuple(round(coord, self.rounding) for coord in g[:3, 3].flatten())
                    valid_poses.append(end_effector_pose)
            
            # Only one thread writes to the all_valid_poses at a time
            with lock:
                all_valid_poses.extend(valid_poses)
                
        lock = threading.Lock()
        
        for t in range(num_threads):
            thread = threading.Thread(target=thread_function, args=(thetas_work[t],))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        self.valid_poses = all_valid_poses
        return all_valid_poses
        
        
    def plot_reachability(self):
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

        plt.show()
        
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
        
        
if __name__ == "__main__":
    
    # Create a database
    from timor.utilities.file_locations import get_module_db_files
    modules_file = get_module_db_files('geometric_primitive_modules')
    db = ModulesDB.from_json_file(modules_file) 

    modules = ('base', 'i_30', 'J2', 'J2', 'J2', 'i_30', 'eef')
    B = ModuleAssembly.from_serial_modules(db, modules)
    long_robot = B.to_pin_robot() #convert to pinocchio robot
    viz = long_robot.visualize()

    how_many_times_to_split_angle_range = 40
    world_resolution = 0.01
    world_dimension = [1.00, 1.00, 1.00]
    num_threads = 5
    
    reachability = Reachability(robot=long_robot, angle_interval=how_many_times_to_split_angle_range, world_resolution=world_resolution)
    
    start_t = time.time()
    valid_poses = reachability.find_all_valid_poses_multithreading(num_threads)
    print(f"Time to find reachability: {time.time() - start_t} seconds")
    
    reachability.plot_reachability()