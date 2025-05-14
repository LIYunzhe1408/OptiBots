import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objs as go
import datetime
import time
import sys
import os
import numpy as np
from timor.Geometry import Box, ComposedGeometry, Cylinder
from timor.task.Obstacle import Obstacle
from timor.task.Task import Task, TaskHeader
from timor.utilities.transformation import Transformation
from timor.utilities.spatial import rotX, rotY, rotZ

# Function to generate cuboid vertices
def cuboid_data(origin, size):
    """ Create cuboid vertices based on the origin and size. """
    x = [origin[0], origin[0] + size[0]]
    y = [origin[1], origin[1] + size[1]]
    z = [origin[2], origin[2] + size[2]]
    return np.array([[x[0], y[0], z[0]],
                    [x[1], y[0], z[0]],
                    [x[1], y[1], z[0]],
                    [x[0], y[1], z[0]],
                    [x[0], y[0], z[1]],
                    [x[1], y[0], z[1]],
                    [x[1], y[1], z[1]],
                    [x[0], y[1], z[1]]])

# Function to plot cuboid
def plot_cuboid(ax, cuboid):
    """ Plot cuboid using its vertices. """
    vertices = [[cuboid[0], cuboid[1], cuboid[2], cuboid[3]],
                [cuboid[4], cuboid[5], cuboid[6], cuboid[7]],
                [cuboid[0], cuboid[1], cuboid[5], cuboid[4]],
                [cuboid[2], cuboid[3], cuboid[7], cuboid[6]],
                [cuboid[1], cuboid[2], cuboid[6], cuboid[5]],
                [cuboid[4], cuboid[7], cuboid[3], cuboid[0]]]

    faces = Poly3DCollection(vertices, linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(faces)

# Function to check for overlap between two cuboids using their axis-aligned bounding boxes (AABB)
def check_overlap(cub1_min, cub1_max, cub2_min, cub2_max):
    """ Check if two cuboids overlap based on their AABBs (axis-aligned bounding boxes). """
    return not (cub1_max[0] <= cub2_min[0] or cub1_min[0] >= cub2_max[0] or
                cub1_max[1] <= cub2_min[1] or cub1_min[1] >= cub2_max[1] or
                cub1_max[2] <= cub2_min[2] or cub1_min[2] >= cub2_max[2])

# Random cuboid volume generation
def random_cuboid_volume(total_volume, num_cuboids):
    volumes = np.random.rand(num_cuboids)
    volumes = volumes / np.sum(volumes) * total_volume
    return volumes

# Function to generate cuboid size, ensuring it fits within the space and respects a maximum size
def generate_random_cuboid(vol, max_size, space_size=(1.0, 1.0, 1.0)):
    """ Generates random cuboid dimensions, ensuring that no dimension exceeds the maximum size. """
    dims = np.random.rand(3)
    dims = dims / np.prod(dims) * vol  # Ensure the product equals the cuboid volume
    dims = np.cbrt(dims)               # Adjust dimensions to ensure valid scaling
    dims = np.minimum(dims, max_size)  # Apply maximum size constraint to each dimension
    dims = np.minimum(dims, space_size)  # Ensure dimensions fit within the space
    return dims

# Plot cuboids and ensure no overlap
def plot_random_cuboids(num_cuboids=20, total_volume=0.001, space_size=(1.0, 1.0, 1.0), max_size=(0.5, 0.5, 0.5), seed=42):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cuboids = []
    volumes = random_cuboid_volume(total_volume, num_cuboids)
    attempts = 0

    # List to store sizes and origins
    cuboid_info = []
    
    x_size, y_size, z_size = space_size
    x_bounds = (-x_size/2.0, x_size/2.0)
    y_bounds = (-y_size/2.0, y_size/2.0)
    z_bounds = (0, z_size)

    # np.random.seed(seed)

    while len(cuboids) < num_cuboids and attempts < 10000:
        vol = volumes[len(cuboids)]
        dims = generate_random_cuboid(vol, max_size, space_size)

        # Generate random position ensuring the cuboid is within space boundaries
        origin = np.array([np.random.uniform(x_bounds[0], x_bounds[1] - dims[0]),
                        np.random.uniform(y_bounds[0], y_bounds[1] - dims[1]),
                        np.random.uniform(z_bounds[0], z_bounds[1] - dims[2])])

        # Calculate min and max corners for the AABB
        cub_min = origin
        cub_max = origin + dims

        # Check for overlap with all previously placed cuboids
        if all(not check_overlap(cub_min, cub_max, cub[0], cub[1]) for cub in cuboids):
            cuboids.append((cub_min, cub_max))
            cuboid = cuboid_data(cub_min, dims)
            plot_cuboid(ax, cuboid)
            # Store size and origin in the desired format
            cuboid_info.append({
                'size': {'x': float(dims[0]), 'y': float(dims[1]), 'z': float(dims[2])},
                'origin': {'x': float(origin[0]), 'y': float(origin[1]), 'z': float(origin[2])}
            })
        attempts += 1
        if attempts >= 10000:
            print("Maximum number of attempts reached. Could not place all cuboids without overlap.")
            break
    plot_cuboid(ax, cuboid_data([-0.5, -0.5, -1.01], [1, 1, 1]))
    cuboid_info.append({'size': {'x': 1, 'y': 1, 'z': 1}, 'origin': {'x': -0.5, 'y': -0.5, 'z': -1.01}}) #create floor

    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim((-z_size, z_size))
    plt.show()

    return cuboid_info

# Helper function to create cuboid edges for Plotly
def create_cuboid_trace(cuboid):
    origin = cuboid['origin']
    size = cuboid['size']
    x, y, z = origin['x'], origin['y'], origin['z']
    dx, dy, dz = size['x'], size['y'], size['z']

    # Define the 8 corners of the cuboid
    corners = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y + dy, z],
        [x + dx, y, z + dz],
        [x, y + dy, z + dz],
        [x + dx, y + dy, z + dz],
    ])

    # Define the 12 edges of the cuboid
    edges = [
        [corners[i] for i in [0, 1, 4, 2, 0]],  # bottom face
        [corners[i] for i in [3, 5, 7, 6, 3]],  # top face
        [corners[i] for i in [0, 3]],           # side edges
        [corners[i] for i in [1, 5]],           
        [corners[i] for i in [2, 6]],           
        [corners[i] for i in [4, 7]]           
    ]

    # Add each edge as a scatter trace
    cuboid_trace = [
        go.Scatter3d(
            x=[edge[i][0] for i in range(len(edge))],
            y=[edge[i][1] for i in range(len(edge))],
            z=[edge[i][2] for i in range(len(edge))],
            mode="lines",
            line=dict(color="blue", width=2)
        )
        for edge in edges
    ]

    return cuboid_trace

# Interactive function to plot cuboids and points
def plot_random_cuboids_with_reachability(cuboid_info, reachable_points=[], unreachable_points=[]):
    fig_data = []  # List to collect all Plotly traces

    # Add cuboid traces based on cuboid_info
    for cuboid in cuboid_info:
        fig_data.extend(create_cuboid_trace(cuboid))

    # Create scatter plot for reachable and unreachable points
    if len(reachable_points) > 0:
        reachable_points = np.array(reachable_points)
        reachable_trace = go.Scatter3d(
            x=reachable_points[:, 0],
            y=reachable_points[:, 1],
            z=reachable_points[:, 2],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='Reachable'
        )
        fig_data.append(reachable_trace)

    if len(unreachable_points) > 0:
        unreachable_points = np.array(unreachable_points)
        unreachable_trace = go.Scatter3d(
            x=unreachable_points[:, 0],
            y=unreachable_points[:, 1],
            z=unreachable_points[:, 2],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.1),
            name='Unreachable'
        )
        fig_data.append(unreachable_trace)

    # Set up the layout for the Plotly figure
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        title='3D Cuboid and Reachability Plot'
    )

    # Create and display the interactive Plotly figure
    fig = go.Figure(data=fig_data, layout=layout)
    fig.show()
    return reachable_points, unreachable_points



def plot_reachability_interactive(reachable_points=[], unreachable_points=[]):

    fig_data = []  # List to collect all Plotly traces

    # Create scatter plot for reachable and unreachable points
    if len(reachable_points) > 0:
        reachable_points = np.array(reachable_points)
        reachable_trace = go.Scatter3d(
            x=reachable_points[:, 0],
            y=reachable_points[:, 1],
            z=reachable_points[:, 2],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='Reachable'
        )
        fig_data.append(reachable_trace)

    if len(unreachable_points) > 0:
        unreachable_points = np.array(unreachable_points)
        unreachable_trace = go.Scatter3d(
            x=unreachable_points[:, 0],
            y=unreachable_points[:, 1],
            z=unreachable_points[:, 2],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.1),
            name='Unreachable'
        )
        fig_data.append(unreachable_trace)

    # Set up the layout for the Plotly figure
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        title='3D Reachability and Cuboid Plot'
    )

    # Create and display the interactive Plotly figure
    fig = go.Figure(data=fig_data, layout=layout)
    fig.show()

def create_random_task():
    cuboid_data_list = plot_random_cuboids(total_volume=0.03, space_size=(1.0, 1.0, 1.0), num_cuboids=2)

    header = TaskHeader(
        ID='Random Obstacles Generation',
        tags=['Capstone', 'demo'],
        date=datetime.datetime(2024, 10, 28),
        author=['Jonas Li'],
        email=['liyunzhe.jonas@berkeley.edu'],
        affiliation=['UC Berkeley']
    )

    box = []
    for idx, info in enumerate(cuboid_data_list):
        size, displacement = info["size"], info["origin"]
        box.append(Obstacle(ID=str(idx), 
                    collision=Box(
                        dict(x=size['x'], y=size['y'], z=size['z']),  # Size
                        pose=Transformation.from_translation([displacement['x'] + size['x'] / 2, displacement['y'] + size['y'] / 2, displacement['z'] + size['z'] / 2])
                    )))# Displacement
    task = Task(header, obstacles=[i for i in box])
    return task
