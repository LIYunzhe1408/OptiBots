import numpy as np
from urdfpy import URDF, Link, Joint, Visual, Geometry, Mesh, Inertial, Collision, JointLimit, Material
from scipy.spatial.transform import Rotation as R


'''
motor4305_link
'''
# Define a 4x4 identity matrix with translation for origin
# Set Up
import os
stl_motor4305_link = "../assets/Assem_4305_JOINT/Assem_4305_JOINT/meshes/motor4305_link.STL"
origin_matrix = np.eye(4)
origin_matrix[:3, 3] = [0, 0, 0]  # translation part

# Inertial
motor4305_link_inertial_origin_matrix = np.eye(4)
motor4305_link_inertial_origin_matrix[:3, 3] = [0.0042941, 1.4489E-06, 0.034665]
motor4305_link_inertial  = Inertial(
    origin=motor4305_link_inertial_origin_matrix,
    mass=0.48076,  # mass in kg
    inertia=np.array([[0.0004787, 5.7034E-09, -5.2375E-06],
                      [5.7034E-09, 0.000333, 1.5111E-08],
                      [-5.2375E-06, 1.5111E-08, 0.00023888]])
)

# visual geometry
motor4305_link_visual = Visual(
    origin=origin_matrix,
    geometry=Geometry(mesh=Mesh(filename=stl_motor4305_link)),
    # material=Material(color=np.array([0.89804, 0.91765, 0.92941, 1]))
)

# collision geometry
motor4305_link_collision = Collision(
    name='',
    geometry=Geometry(mesh=Mesh(filename=stl_motor4305_link)),
    origin=origin_matrix
)


motor4305_link = Link(
    name="motor4305_link",  # Child link for the joint
    inertial=motor4305_link_inertial,
    visuals=[motor4305_link_visual],
    collisions=[motor4305_link_collision]
)



'''
motor4305_out_link
'''
# Define a 4x4 identity matrix with translation for origin
# Set Up
stl_motor4305_out_link = "../assets/Assem_4305_JOINT/Assem_4305_JOINT/meshes/motor4305_out_link.STL"
origin_matrix = np.eye(4)
origin_matrix[:3, 3] = [0, 0, 0]  # translation part

# Inertial
motor4305_out_link_inertial_origin_matrix = np.eye(4)
motor4305_out_link_inertial_origin_matrix[:3, 3] = [-1.3533E-08, -0.0025, -3.5182E-09]
motor4305_out_link_inertial = Inertial(
    origin=motor4305_out_link_inertial_origin_matrix,
    mass=0.026746,  # mass in kg
    inertia=np.array([[5.4686E-06, 4.832E-22, -3.7058E-22],
                      [4.832E-22, 1.0826E-05, -3.2034E-22],
                      [-3.7058E-22, -3.2034E-22, 5.4686E-06]])
)

# visual geometry
motor4305_out_link_visual = Visual(
    geometry=Geometry(mesh=Mesh(filename=stl_motor4305_out_link)),
    origin=origin_matrix,
    # material=Material(color=[0.89804, 0.91765, 0.92941, 1])
)

# collision geometry
motor4305_out_link_collision = Collision(
    name='',
    geometry=Geometry(mesh=Mesh(filename=stl_motor4305_out_link)),
    origin=origin_matrix
)


motor4305_out_link = Link(
    name="motor4305_out_link",  # Child link for the joint
    inertial=motor4305_out_link_inertial,
    visuals=[motor4305_out_link_visual],
    collisions=[motor4305_out_link_collision]
)

'''
Joint
'''
limit = JointLimit(
    lower=0,  # minimum rotation in rad
    upper=0,   # maximum rotation in rad
    effort=0,   # ?
    velocity=0  # max velocity
)

# Define rotation (roll, pitch, yaw) in radians
rpy = [0, 0.032992, 1.5708]
joint_origin_matrix = np.eye(4)

# Create rotation matrix from RPY
rotation_matrix = R.from_euler('xyz', rpy).as_matrix()
joint_origin_matrix[:3, :3] = rotation_matrix
joint_origin_matrix[:3, 3] = [0.0255, -0.0395, 0]
joint = Joint(
    name="motor4305_rev_joint",
    joint_type="revolute",
    origin=joint_origin_matrix,
    parent="motor4305_link",
    child="motor4305_out_link",
    axis=(0, 0, 1),
    limit=limit
)


robot = URDF(
    name="Assem_4305_JOINT",
    links=[motor4305_link, motor4305_out_link],
    joints=[joint]
)

# Write the URDF model to a file
robot.save('./Assem_4305_JOINT.urdf')


"""
Run HERE
"""
usePredefined = True

if usePredefined:
    # Use Pre-defined URDF
    robot = URDF.load('../assets/Assem_4305_JOINT/Assem_4305_JOINT/urdf/Assem_4305_JOINT.urdf')
else:
    # Use converted URDF
    robot = URDF.load('./Assem_4305_JOINT.urdf')


robot.animate()
