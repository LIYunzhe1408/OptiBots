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

from util import generate_header, read_rod_trans, fetch_joint_trans

ROT_X = Transformation.from_rotation(rotX(np.pi)[:3, :3])
ROT_Z = Transformation.from_rotation(rotZ(np.pi)[:3, :3])
ROT_Y = Transformation.from_rotation(rotY(np.pi)[:3, :3])

ROT_X_90 = Transformation.from_rotation(rotX(np.pi/2)[:3, :3])
ROT_Z_90 = Transformation.from_rotation(rotZ(np.pi/2)[:3, :3])
ROT_Y_90 = Transformation.from_rotation(rotY(np.pi/2)[:3, :3])
DIAMETER = 80 / 1000

def create_eef() -> ModulesDB:
    """Creates a simplified end effector module."""
    geometry = Sphere({'r': DIAMETER / 6}, pose=Transformation.from_translation([0, 0, DIAMETER / 2]))
    c_robot = Connector('robot2eef', gender=Gender.f, connector_type='default', size=[25 / 1000, 25 / 1000],
                        body2connector=ROT_X @ Transformation.from_translation([0, -DIAMETER / 3, -DIAMETER / 2]))
    c_world = Connector('end-effector', gender=Gender.m, connector_type='eef', size=[None,None],
                        body2connector=Transformation.from_translation([0, 0, DIAMETER / 2]))
    return ModulesDB({
        AtomicModule(generate_header("eef", "End Effector: eef"),
                     [Body('EEF', collision=geometry, connectors=[c_robot, c_world])], [])
    })

# def i_links() -> ModulesDB:
#     """For every size, creates an I-shaped link (aka a cylinder) with two connectors."""
#     sizes = (150 / 1000, 300 / 1000, 450 / 1000)
#     diameter = 25 / 1000
#     links = ModulesDB()
#     for size in sizes:
#         module_header = ModuleHeader(ID='i_{}'.format(int(size * 100)),
#                                      name='I shaped link {}-{}-{}'.format(diameter, diameter, int(size * 100)),
#                                     date="2024-12-05",
#                                     author=['Jonathan Külz'],
#                                     email=['jonathan.kuelz@tum.de'],
#                                     affiliation=['Technical University of Munich']
#                                     )
#         connectors = (
#             Connector(f'{int(diameter * 100)}-{i}',
#                     ROT_X @ Transformation.from_translation([0, 0, size / 2]) if i == 0
#                     else Transformation.from_translation([0, 0, size / 2]),
#                     gender=Gender.f if i == 0 else Gender.m,
#                     connector_type='default',
#                     size=[diameter, diameter])
#             for i in range(2)
#         )
#         geometry = Box({'x': diameter, 'y': diameter, 'z': size}, pose=Transformation.from_translation([0, 0, 0]))
#         body = Body('i_{}'.format(int(size * 100)), collision=geometry, connectors=connectors,
#                     inertia=square_rod_inertia(size, diameter))
#         links.add(AtomicModule(module_header, [body]))
#     return links

def create_connectors_for_link(length, diameter, trans):
    connectors = []
    for i in range(2):
        connectors.append(Connector(
            f'{int(diameter * 100)}-{i}', 
            trans[i],
            gender=Gender.m if i == 0 else Gender.f,
            connector_type='default', 
            size=[diameter, diameter]))
    return connectors

def create_i_links(rod_name) -> ModulesDB:
    """For every size, creates an I-shaped link (aka a cylinder) with two connectors."""
    sizes = (150 / 1000, 300 / 1000, 450 / 1000)
    diameter = 25 / 1000
    links = ModulesDB()
    
    
    for size in sizes:
        rod_id = f'{rod_name}-{size}'
        trans = read_rod_trans(rod_name, size, diameter)
        header = generate_header(rod_id, rod_id)
        # connectors = (
        #     Connector(rod_id,
        #             ROT_X @ Transformation.from_translation([0, size/2, 0]) if i == 0
        #             else Transformation.from_translation([0, 0, 0.077]),
        #             gender=Gender.f if i == 0 else Gender.m,
        #             connector_type='default',
        #             size=[diameter, diameter])
        #     for i in range(2)
        # )
        # print(type(connectors))
        connectors = create_connectors_for_link(size, diameter, trans)
        geometry = Box({'x': diameter, 'y': diameter, 'z': size}, pose=Transformation.from_translation([0, 0, 0]))
        body = Body(rod_id, collision=geometry, connectors=connectors,
                    inertia=square_rod_inertia(size, diameter))
        links.add(AtomicModule(header, [body]))
    return links

def generate_i_links(base, joints):
    sizes = (100/1000, 150 / 1000, 200/1000, 300 / 1000)
    diameter = 25 / 1000
    directions = ['N', 'E', 'W', 'S']
    ROT_Z_90 = Transformation.from_rotation(rotZ(np.pi/2)[:3, :3])
    rotate = Transformation.from_rotation(rotZ(0)[:3, :3])
    links = ModulesDB()
    for joint in joints:
        for size in sizes:
            for direction in directions:
                version = 0
                for base_trans in fetch_joint_trans(base.id, size, diameter, Gender.f):
                    for joint_trans in fetch_joint_trans(joint.id, size, diameter, Gender.m):
                        rod_id = base.id + "-to-" + joint.id + "-" + str(size) + "-" + str(version) + "-" + direction
                        header = generate_header(rod_id, rod_id)

                        connectors = create_connectors_for_link(size, diameter, [joint_trans, rotate @ base_trans])
                        geometry = Box({'x': diameter, 'y': diameter, 'z': size}, pose=Transformation.from_translation([0, 0, 0]))
                        body = Body(rod_id, collision=geometry, connectors=connectors,
                                    inertia=square_rod_inertia(size, diameter))
                        #print(rod_id)
                        links.add(AtomicModule(header, [body]))
                        
                        version += 1
                rotate = rotate @ ROT_Z_90

    rotate = Transformation.from_rotation(rotZ(0)[:3, :3])
    for f_joint in joints:
        for m_joint in joints:
            for size in sizes:
                for direction in directions:
                    version = 0
                    for f_joint_trans in fetch_joint_trans(f_joint.id, size, diameter, Gender.f):
                        for m_joint_trans in fetch_joint_trans(m_joint.id, size, diameter, Gender.m):
                            rod_id = f_joint.id + "-to-" + m_joint.id + "-" + str(size) + "-" + str(version) + "-" + direction
                            header = generate_header(rod_id, rod_id)
                            
                            connectors = create_connectors_for_link(size, diameter, [m_joint_trans, rotate @ f_joint_trans])
                            geometry = Box({'x': diameter, 'y': diameter, 'z': size}, pose=Transformation.from_translation([0, 0, 0]))
                            body = Body(rod_id, collision=geometry, connectors=connectors,
                                        inertia=square_rod_inertia(size, diameter))
                            links.add(AtomicModule(header, [body]))
                            version += 1
                    rotate = rotate @ ROT_Z_90
    return links


def base(urdf_path):
    """Creates a base connector attached to a box."""
    dir_name = urdf_path.split('/')[1]
    ROT_X = Transformation.from_rotation(rotX(np.pi)[:3, :3])
    urdf_dict = urdf_to_dict(urdf_path)
    joint = urdf_dict['robot']['joint']
    proximal_name = joint['parent']['link']
    distal_name = joint['child']['link']
    links = urdf_dict['robot']['link']
    directory_path = os.path.dirname(urdf_path)
    
    for link in links:
        link_name = link['name']
        stl_path = link['collision']['geometry']['mesh']['filename']
        assets_path = os.path.join("assets", dir_name, dir_name, stl_path.split('/')[-2], stl_path.split('/')[-1])
        if link_name == proximal_name:
            proximal_inertial = link['inertial']
            proximal_origin = link['collision']['origin']
            proximal_geometry = Mesh({"file": assets_path})
        elif link_name == distal_name:
            distal_inertial = link['inertial']
            distal_origin = link['collision']['origin']
            distal_geometry = Mesh({"file": assets_path})
    
    

    diameter = 25 / 1000
    # r_p, p_p = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in proximal_origin['xyz'].split(" ")], [float(x) for x in proximal_origin['rpy'].split(" ")])
    # r_d, p_d = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in distal_origin['xyz'].split(" ")], [float(x) for x in distal_origin['rpy'].split(" ")])
    
    ROT_X = Transformation.from_rotation(-rotX(np.pi/2)[:3, :3])
    ROT_Y = Transformation.from_rotation(rotY(np.pi/2)[:3, :3])
    ROT_Z = Transformation.from_rotation(rotY(np.pi)[:3, :3])
    ROT_Z_90 = Transformation.from_rotation(rotY(np.pi/2)[:3, :3])
    EYE = Transformation.from_rotation(rotX(0)[:3, :3])

    c_type = 'base' #if  joint['name'] == 'base_rev_joint' else 'default'
    ROTATE_PROXIMAL = ROT_X if c_type == 'base' else EYE
    ROTATE_DISTAL = EYE if c_type == 'base' else EYE

    proximal_connector = Connector(
                                    connector_id=proximal_name+"connector",
                                    body2connector=ROTATE_PROXIMAL @ Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix(np.array([0,0, 0])),       
                                                    # R=r_p,
                                                    p=np.array([0, 0.0, 0]),
                                                    # p=p_p
                                    ),
                                    #body2connector = Transformation.from_translation([0, -0.0, 1]),
                                    #body2connector = Transformation.from_translation([float(x) for x in proximal_origin['xyz'].split(" ")]) if c_type == 'base' else ROT_X@Transformation.from_translation([float(x) for x in proximal_origin['xyz'].split(" ")]),
                                    gender=Gender.f,
                                    connector_type=c_type,
                                    size=[diameter, diameter]
        )
    distal_connector = Connector(
                                    connector_id=distal_name+"connector",
                                    body2connector=ROTATE_DISTAL @ Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix(np.array([0,0, 0])),
                                                    p=np.array([0, -0.0, 0.0]),            
                                                    # R=r_d,
                                                    # p=p_d
                                    ),
                                    # body2connector = Transformation.from_translation([0, -1.0, -1]),
                                    #body2connector = Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]) if c_type == 'base' else ROT_X@Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]),
                                    gender=Gender.m,
                                    connector_type='default',
                                    size=[diameter, diameter]
                                    
        )            
    
    proximal = Body(proximal_name, collision=proximal_geometry,
                    connectors=[proximal_connector],
                    )
    distal = Body(distal_name, collision=distal_geometry,
                    connectors=[distal_connector],
                    )
    
    r_joint = Joint(
        joint_id=urdf_dict['robot']['name'],
        joint_type=joint['type'],
        parent_body=proximal,
        child_body=distal,
        q_limits=np.array([-np.pi, np.pi]) if joint['name'] == 'revolute' else (-np.inf, np.inf),
        torque_limit=1000,
        acceleration_limit=5,
        velocity_limit=10,
        parent2joint=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]),
                                                    p=joint['origin']['xyz'].split(" ")
                                ),
        joint2child=Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix(np.array([0, 0, 0])),
                                                    p=np.array([0, -0.0, 0])
                                )
    )
    return AtomicModule(generate_header(urdf_dict['robot']['name'], 'Revolute Joint: ' + joint['name']), [proximal, distal], [r_joint])
    
    # return ModulesDB({
    #     AtomicModule(ModuleHeader(ID='base', name='Base', author=['Jonathan Külz'], date="2024-12-03",
    #                             email=['jonathan.kuelz@tum.de'], affiliation=['Technical University of Munich']),
    #                 [Body('base', collision=c_body, connectors=[c_world, c_robot])])
    # })
def create_revolute_joint(urdf_path: str):
    dir_name = urdf_path.split('/')[1]
    urdf_dict = urdf_to_dict(urdf_path)
    module_name = urdf_dict['robot']['name'].split(".")[0]
    joint = urdf_dict['robot']['joint']
    proximal_name = joint['parent']['link']
    distal_name = joint['child']['link']
    links = urdf_dict['robot']['link']

    
    for link in links:
        link_name = link['name']
        stl_path = link['collision']['geometry']['mesh']['filename']
        assets_path = os.path.join("assets", dir_name, dir_name, stl_path.split('/')[-2], stl_path.split('/')[-1])
        if link_name == proximal_name:
            proximal_inertial = link['inertial']
            proximal_origin = link['collision']['origin']
            proximal_geometry = Mesh({"file": assets_path})
        elif link_name == distal_name:
            distal_inertial = link['inertial']
            distal_origin = link['collision']['origin']
            distal_geometry = Mesh({"file": assets_path})
    
    

    diameter = 25 / 1000
    # r_p, p_p = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in proximal_origin['xyz'].split(" ")], [float(x) for x in proximal_origin['rpy'].split(" ")])
    # r_d, p_d = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in distal_origin['xyz'].split(" ")], [float(x) for x in distal_origin['rpy'].split(" ")])
    
    ROT_X = Transformation.from_rotation(-rotX(np.pi/2)[:3, :3])
    ROT_Y = Transformation.from_rotation(rotY(np.pi/2)[:3, :3])
    ROT_Z = Transformation.from_rotation(rotZ(np.pi)[:3, :3])
    ROT_Z_90 = Transformation.from_rotation(rotZ(np.pi/2)[:3, :3])
    EYE = Transformation.from_rotation(rotX(0)[:3, :3])

    c_type = 'base' if  'base' in module_name else 'default'
    ROTATE_PROXIMAL = ROT_X @ ROT_X if c_type == 'base' else EYE
    ROTATE_DISTAL = EYE if c_type == 'base' else EYE #Transformation.from_rotation(rotX(np.pi/2)[:3, :3])

    MOVE_P = Transformation.from_translation([0, 0, 0]) if module_name == '540_joint' else Transformation.from_translation([0,0,0])
    MOVE_D = Transformation.from_translation([0, 0, 0]) if module_name == '540_joint' else Transformation.from_translation([0,0,0])
    proximal_connector = Connector(
                                    connector_id=module_name + proximal_name+"connector",
                                    body2connector=ROTATE_PROXIMAL @ Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix(np.array([0,0, 0])),       
                                                    # R=r_p,
                                                    p=np.array([0, 0.0, 0.0]),
                                                    # p=p_p
                                    ),
                                    #body2connector = Transformation.from_translation([float(x) for x in proximal_origin['xyz'].split(" ")]) if c_type == 'base' else ROT_X@Transformation.from_translation([float(x) for x in proximal_origin['xyz'].split(" ")]),
                                    gender=Gender.f,
                                    connector_type=c_type,
                                    size=[diameter, diameter]
        )
    distal_connector = Connector(
                                    connector_id=module_name + distal_name+"connector",
                                    body2connector=ROTATE_DISTAL @ Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix(np.array([0,0, 0])),
                                                    p=np.array([0, 0.0, 0]),            
                                                    # R=r_d,
                                                    # p=p_d
                                    ),
                                    #body2connector = Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]) if c_type == 'base' else ROT_X@Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]),
                                    gender=Gender.m,
                                    connector_type='default',
                                    size=[diameter, diameter]
                                    
        )            
    
    proximal = Body(module_name + proximal_name, collision=proximal_geometry,
                    connectors=[proximal_connector],
                    inertia=create_inertia(proximal_inertial)
                    )
    distal = Body(module_name + distal_name, collision=distal_geometry,
                    connectors=[distal_connector],
                    inertia=create_inertia(distal_inertial)
                    )
    
    ROTATE_JOINT_P = rpy_to_rotation_matrix([np.pi/2, 0, 0])
    ROTATE_JOINT_C = rpy_to_rotation_matrix([-np.pi/2, 0, 0])
    ROTATE_JOINT_P = rpy_to_rotation_matrix([0, 0, 0]) if module_name != '540_joint' else rpy_to_rotation_matrix([np.pi/2, 0, 0])
    ROTATE_JOINT_C = rpy_to_rotation_matrix([0, 0, 0]) if module_name != '540_joint' else rpy_to_rotation_matrix([-np.pi/2, 0, 0])
    MOVE_JOINT_P = Transformation.from_translation([0, 0, diameter*1.2]) if module_name == '540_joint' else Transformation.from_translation([0,0,0])
    MOVE_JOINT_C = Transformation.from_translation([0, -diameter*1.2, 0]) if module_name == '540_joint' else Transformation.from_translation([0,0,0])
    
    r_joint = Joint(
        joint_id=module_name,
        joint_type=joint['type'],
        parent_body=proximal,
        child_body=distal,
        q_limits=np.array([-np.pi, np.pi]) if joint['type'] == 'revolute' else (-np.inf, np.inf),
        torque_limit=1000,
        acceleration_limit=5,
        velocity_limit=10,
        parent2joint=MOVE_JOINT_P @Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]) @ ROTATE_JOINT_P,
                                                    p=joint['origin']['xyz'].split(" ")
                                ),
        joint2child=MOVE_JOINT_C @ Transformation.from_roto_translation(
                                                    R=rpy_to_rotation_matrix(np.array([0, 0, 0])) @ ROTATE_JOINT_C,
                                                    p=np.array([0, -0.0, 0])
                                )
    )
    # if c_type == 'base':
    #     return AtomicModule(generate_header(joint['name'], 'Revolute Joint: ' + joint['name']), [proximal, distal],[])
    
    return AtomicModule(generate_header(module_name, joint['type'] + " joint: " + module_name), [proximal, distal], [r_joint])

# def create_revolute_joint(urdf_path) -> ModulesDB:
#     """Creates an L-shaped joint"""
#     dir_name = urdf_path.split('/')[1]
#     urdf_dict = urdf_to_dict(urdf_path)
#     joint = urdf_dict['robot']['joint']
#     proximal_name = joint['parent']['link']
#     distal_name = joint['child']['link']
#     links = urdf_dict['robot']['link']
#     directory_path = os.path.dirname(urdf_path)
    
#     # for link in links:
#     #     if link['name'] == proximal_name:
#     #         proximal_serialized_geometry = {
#     #             "type": "mesh",  # Type of geometry
#     #             "parameters": {
#     #                 "file": link['collision']['geometry']['mesh']['filename']  # Path to the STL file
#     #             }
#     #         }
#     #         proximal_body = Geometry.from_json_data([proximal_serialized_geometry], directory_path)
#     #         proximal_inertial = link['inertial']
#     #         proximal_origin = link['collision']['origin']
#     #     elif link['name'] == distal_name:
#     #         distal_serialized_geometry = {
#     #             "type": "mesh",  # Type of geometry
#     #             "parameters": {
#     #                 "file": link['collision']['geometry']['mesh']['filename']  # Path to the STL file
#     #             }
#     #         }
#     #         distal_body = Geometry.from_json_data([distal_serialized_geometry], directory_path)
#     #         distal_inertial = link['inertial']
#     #         distal_origin = link['collision']['origin']
#     for link in links:
#         link_name = link['name']
#         stl_path = link['collision']['geometry']['mesh']['filename']
#         assets_path = os.path.join("assets", dir_name, dir_name, stl_path.split('/')[1], stl_path.split('/')[2])
#         if link_name == proximal_name:
#             proximal_inertial = link['inertial']
#             proximal_origin = link['collision']['origin']
#             proximal_geometry = Mesh({"file": assets_path})
#         elif link_name == distal_name:
#             distal_inertial = link['inertial']
#             distal_origin = link['collision']['origin']
#             distal_geometry = Mesh({"file": assets_path})

#     length = 150 / 1000
#     diameter = 25 / 1000
#     # proximal_body = ComposedGeometry((Cylinder({'r': diameter / 2, 'z': length}),
#     #                                 Sphere({'r': diameter / 2}, pose=Transformation.from_translation([0, 0, length / 2])))
#     #                                 )
#     # distal_body = Cylinder({'r': diameter / 2, 'z': 150 / 1000})
#     r_p, p_p = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in proximal_origin['xyz'].split(" ")], [float(x) for x in proximal_origin['rpy'].split(" ")])
#     r_d, p_d = body2connector_helper([float(x) for x in joint['origin']['xyz'].split(" ")], [float(x) for x in joint['origin']['rpy'].split(" ")], [float(x) for x in distal_origin['xyz'].split(" ")], [float(x) for x in distal_origin['rpy'].split(" ")])
#     c_type = 'base' if  joint['name'] == 'base_rev_joint' else 'default'
    
#     print([float(x) for x in proximal_origin['xyz'].split(" ")])
#     proximal_connector = Connector(
#                                     connector_id=proximal_name+"connector",
#                                     # body2connector=Transformation.from_roto_translation(
#                                     #                 R=ROT_X,       
#                                     #                 #R=r_p,
#                                     #                 p=np.array([float(x) for x in proximal_origin['xyz'].split(" ")]),
#                                     #                 #p=p_p,
#                                     # ),
#                                     body2connector = Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]),
#                                     gender=Gender.f,
#                                     connector_type=c_type,
#                                     size=[diameter, diameter]
#         )
#     distal_connector = Connector(
#                                     connector_id=distal_name+"connector",
#                                     # body2connector=Transformation.from_roto_translation(
#                                     #                 R=rpy_to_rotation_matrix(np.array([0,0, 0])),
#                                     #                 p=np.array([0, -0.0, 0]),            
#                                     #                 # R=r_d,
#                                     #                 # p=p_d
#                                     # ),
#                                     body2connector = Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]) if c_type == 'base' else Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]),
#                                     gender=Gender.m, #if not eef else Gender.h,
#                                     connector_type='default',
#                                     size=[diameter, diameter]
                                    
#         )            
    
#     proximal = Body(proximal_name, collision=proximal_geometry,
#                     connectors=[proximal_connector],
#                     inertia=create_inertia(proximal_inertial)
#                     )
#     distal = Body(distal_name, collision=distal_geometry,
#                     connectors=[distal_connector],
#                     inertia=create_inertia(distal_inertial)
#                     )
    
#     r_joint = Joint(
#         joint_id=joint['name'],
#         joint_type=joint['type'],
#         parent_body=proximal,
#         child_body=distal,
#         q_limits=np.array([-np.pi, np.pi]),
#         torque_limit=1000,
#         acceleration_limit=5,
#         velocity_limit=10,
#         parent2joint=Transformation.from_roto_translation(
#                                                     R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['rpy'].split(" ")]),
#                                                     p=joint['origin']['xyz'].split(" ")
#                                 ),
#         joint2child=Transformation.from_roto_translation(
#                                                     R=rpy_to_rotation_matrix([float(x) for x in joint['origin']['xyz'].split(" ")]),
#                                                     p=np.array([0, -0.0, 0])
#                                 )
#         #parent2joint=Transformation.from_translation([float(x) for x in proximal_origin['xyz'].split(" ")]),
#         #joint2child=Transformation.from_translation([float(x) for x in distal_origin['xyz'].split(" ")]),
#         #parent2joint=Transformation.from_translation([0,0,0.5]),
#         #joint2child=Transformation.from_translation([0,0,0]),
#     )
#     return AtomicModule(generate_header(joint['name'], 'Revolute Joint: ' + joint['name']), [proximal, distal], [r_joint])
#     return ModulesDB({
#         AtomicModule(module_header, [proximal, distal], [r_joint])
#     })
if __name__ == "__main__":
    print("hello")
    # urdf_file = "assets/Assem_4305_JOINT/Assem_4305_JOINT/urdf/Assem_4305_JOINT.urdf"
    # r_joint = revolute_joint(urdf_file)
    # r_joint.debug_visualization()