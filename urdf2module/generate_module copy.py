from util import urdf_to_dict, rpy_to_rotation_matrix, create_inertia, square_rod_mass, square_rod_inertia, body2connector_helper
from datetime import datetime
from pathlib import Path

import numpy as np
import os
import pinocchio as pin

from timor.Geometry import Geometry
from timor.Bodies import Body, Connector, Gender
from timor.Joints import Joint
from timor.Module import AtomicModule, ModulesDB, ModuleHeader
from timor.Geometry import Box, ComposedGeometry, Cylinder, Sphere, Mesh
from timor.utilities.transformation import Transformation
from timor.utilities.spatial import rotX, rotY, rotZ


def i_links() -> ModulesDB:
    """For every size, creates an I-shaped link (aka a cylinder) with two connectors."""
    ROT_X = Transformation.from_rotation(rotX(np.pi)[:3, :3])
    sizes = (150 / 1000, 300 / 1000, 450 / 1000)
    diameter = 25 / 1000
    links = ModulesDB()
    for size in sizes:
        module_header = ModuleHeader(ID='i_{}'.format(int(size * 100)),
                                     name='I shaped link {}-{}-{}'.format(diameter, diameter, int(size * 100)),
                                    date="2024-12-05",
                                    author=['Jonathan Külz'],
                                    email=['jonathan.kuelz@tum.de'],
                                    affiliation=['Technical University of Munich']
                                    )
        connectors = (
            Connector(f'{int(diameter * 100)}-{i}',
                    ROT_X @ Transformation.from_translation([0, 0, size / 2]) if i == 0
                    else Transformation.from_translation([0, 0, size / 2]),
                    gender=Gender.f if i == 0 else Gender.m,
                    connector_type='default',
                    size=[diameter, diameter])
            for i in range(2)
        )
        geometry = Box({'x': diameter, 'y': diameter, 'z': size}, pose=Transformation.from_translation([0, 0, 0]))
        body = Body('i_{}'.format(int(size * 100)), collision=geometry, connectors=connectors,
                    inertia=square_rod_inertia(size, diameter))
        links.add(AtomicModule(module_header, [body]))
    return links

def base(urdf_path) -> ModulesDB:
    """Creates a base connector attached to a box."""
    ROT_X = Transformation.from_rotation(rotX(np.pi)[:3, :3])
    urdf_dict = urdf_to_dict(urdf_path)
    joint = urdf_dict['robot']['joint']
    proximal_name = joint['parent']['link']
    distal_name = joint['child']['link']
    links = urdf_dict['robot']['link']
    directory_path = os.path.dirname(urdf_path)
    
    for link in links:
        if link['name'] == proximal_name:
            proximal_serialized_geometry = {
                "type": "mesh",  # Type of geometry
                "parameters": {
                    "file": link['collision']['geometry']['mesh']['filename']  # Path to the STL file
                }
            }
            c_body = Geometry.from_json_data([proximal_serialized_geometry], directory_path)
            proximal_inertial = link['inertial']
        elif link['name'] == distal_name:
            distal_serialized_geometry = {
                "type": "mesh",  # Type of geometry
                "parameters": {
                    "file": link['collision']['geometry']['mesh']['filename']  # Path to the STL file
                }
            }
            c_body_2 = Geometry.from_json_data([distal_serialized_geometry], directory_path)
            distal_inertial = link['inertial']
    
    length = 150 / 1000
    diameter = 25 / 1000
    
    c_world = Connector('base', ROT_X, gender=Gender.f, connector_type='base', size=[diameter, diameter])
    c_robot = Connector('base2robot', gender=Gender.m, connector_type='default', size=[diameter, diameter],
                        body2connector=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]),
                                                    p=joint['origin']['xyz'].split(" ")
                                ))
    return AtomicModule(ModuleHeader(ID='base', name='Base', author=['Jonathan Külz'], date="2024-12-03",
                                email=['jonathan.kuelz@tum.de'], affiliation=['Technical University of Munich']),
                    [Body('base', collision=c_body, connectors=[c_world, c_robot])])
    
    # return ModulesDB({
    #     AtomicModule(ModuleHeader(ID='base', name='Base', author=['Jonathan Külz'], date="2024-12-03",
    #                             email=['jonathan.kuelz@tum.de'], affiliation=['Technical University of Munich']),
    #                 [Body('base', collision=c_body, connectors=[c_world, c_robot])])
    # })
    
def revolute_joint(urdf_path) -> ModulesDB:
    """Creates an L-shaped joint"""
    urdf_dict = urdf_to_dict(urdf_path)
    joint = urdf_dict['robot']['joint']
    proximal_name = joint['parent']['link']
    distal_name = joint['child']['link']
    links = urdf_dict['robot']['link']
    directory_path = os.path.dirname(urdf_path)
    
    for link in links:
        if link['name'] == proximal_name:
            proximal_serialized_geometry = {
                "type": "mesh",  # Type of geometry
                "parameters": {
                    "file": link['collision']['geometry']['mesh']['filename']  # Path to the STL file
                }
            }
            proximal_body = Geometry.from_json_data([proximal_serialized_geometry], directory_path)
            proximal_inertial = link['inertial']
            proximal_origin = link['collision']['origin']
        elif link['name'] == distal_name:
            distal_serialized_geometry = {
                "type": "mesh",  # Type of geometry
                "parameters": {
                    "file": link['collision']['geometry']['mesh']['filename']  # Path to the STL file
                }
            }
            distal_body = Geometry.from_json_data([distal_serialized_geometry], directory_path)
            distal_inertial = link['inertial']
            distal_origin = link['collision']['origin']

    length = 150 / 1000
    diameter = 25 / 1000
    # proximal_body = ComposedGeometry((Cylinder({'r': diameter / 2, 'z': length}),
    #                                 Sphere({'r': diameter / 2}, pose=Transformation.from_translation([0, 0, length / 2])))
    #                                 )
    # distal_body = Cylinder({'r': diameter / 2, 'z': 150 / 1000})
    r_p, p_p = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in proximal_origin['xyz'].split(" ")], [float(x) for x in proximal_origin['rpy'].split(" ")])
    r_d, p_d = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in distal_origin['xyz'].split(" ")], [float(x) for x in distal_origin['rpy'].split(" ")])
    proximal_connector = Connector(
                                connector_id=proximal_name+"connector",
                                body2connector=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]),
                                                    p=joint['origin']['xyz'].split(" ")
                                ),
                                gender=Gender.m,
                                connector_type='default',
                                size=[diameter, diameter]
                                
    )
    distal_connector = Connector(
                                connector_id=distal_name+"connector",
                                body2connector=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]),
                                                    p=joint['origin']['xyz'].split(" ")
                                ),
                                # body2connector=Transformation.from_roto_translation(
                                #                     R=rpy_to_rotation_matrix(np.array([0.0255, -0.0395, 0])),
                                #                     p=np.array([0.0255, -0.0395, 0])
                                # ),
                                gender=Gender.f,
                                connector_type='default',
                                size=[diameter, diameter]
                                
    )
    proximal = Body(proximal_name, collision=proximal_body,
                    connectors=[proximal_connector],
                    inertia=create_inertia(proximal_inertial)
                    )
    distal = Body(distal_name, collision=distal_body,
                connectors=[distal_connector],
                inertia=create_inertia(distal_inertial)
                )
    
    r_joint = Joint(
        joint_id=joint['name'],
        joint_type=joint['type'],
        parent_body=proximal,
        child_body=distal,
        q_limits=np.array([-np.pi, np.pi]),
        torque_limit=1000,
        acceleration_limit=5,
        velocity_limit=10,
        parent2joint=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]),
                                                    p=joint['origin']['xyz'].split(" ")
                                ),
        joint2child=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['xyz'].split(" ")]),
                                                    p=np.array([0, -0.0, 0])
                                )
    )
    # print([float(x) for x in joint['origin']['rpy'].split(" ")])
    # print(rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]))
    # print(rpy_to_rotation_matrix(np.array([0.0255, -0.0395, 0])))

    module_header = ModuleHeader(ID=joint['name'],
                                name='Revolute Joint: ' + joint['name'],
                                date="2024-12-03",
                                author=['Jae Won Kim'],
                                email=['jaewon_kim@berkeley.edu'],
                                affiliation=['University of California, Berkeley']
                                )
    return AtomicModule(module_header, [proximal, distal], [r_joint])
    return ModulesDB({
        AtomicModule(module_header, [proximal, distal], [r_joint])
    })
if __name__ == "__main__":
    print("hello")
    # urdf_file = "assets/Assem_4305_JOINT/Assem_4305_JOINT/urdf/Assem_4305_JOINT.urdf"
    # r_joint = revolute_joint(urdf_file)
    # r_joint.debug_visualization()