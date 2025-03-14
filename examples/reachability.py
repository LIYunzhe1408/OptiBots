import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from pathlib import Path
import plotly.graph_objects as go # interactive graphs
from math import ceil

import timor
from timor.utilities import prebuilt_robots
from timor.task.Task import *
from timor.task.Obstacle import *
from timor.Module import *
from timor.utilities.visualization import animation
from timor.Geometry import Box, ComposedGeometry, Cylinder




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
        self.obstacles_found = 0
        
        self.valid_poses = []
        
    
    def reachability_random_sample(self, num_samples):
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
            if self.task and self.robot.has_collisions(task, safety_margin = self.safety_margin):
                continue
                
            end_effector_pose = tuple(round(coord, 2) for coord in tcp_pose[:3, 3].flatten())

            # Compute the Yoshikawa manipulability index
            # High Yoshikawa manipulability index values indicate that the robot has good dexterity 
            # and can move in multiple directions from the corresponding configurations. 
            # Low manipulability index values indicate that corresponding configurations are less 
            # dexterous and more susceptible to singularities.
            manipulability_idk = self.robot.manipulability_index(q)
            reachable_points.add((end_effector_pose, manipulability_idk))  
            
        self.reachable, self.manipulability =  zip([reachable_points])
        

        
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
    
    reachability = Reachability(robot = default_robot, 
                                angle_interval = how_many_times_to_split_angle_range, 
                                world_resolution = world_resolution, 
                                world_dimension = world_dimension,
                                task = task)
    
    start_t = time.time()
    valid_poses = reachability.reachability_random_sample(num_samples = 100000)
    print(f"Time to find reachability: {time.time() - start_t} seconds")
    
    reachable, manipulability =  zip(*valid_poses)
    reachable = np.array([list(pt) for pt in reachable])
    print("reachable raw ", reachable)
    
    reachability.plot_interactive_reachability_with_manipulability(reachable, manipulability)
    percentage = reachability.find_reachibility_percentage(voxel_size = 0.5, valid_pose = reachable)
    
    print(f"percentage {percentage}%")
    
    # percentage = reachability.find_reachibility_percentage()
    # print(f"Percentage of reachability: {percentage}%")
    