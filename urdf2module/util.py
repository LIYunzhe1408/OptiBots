import xml.etree.ElementTree as ET
import numpy as np
import pinocchio
from pprint import pprint
from scipy.spatial.transform import Rotation

def square_rod_mass(l: float, side: float, side_inner: float = 0) -> float:
    """
    Calculates the mass of a solid or hollow square rod.
    
    :param l: Length of the rod (meters).
    :param side: Outer side length of the square cross-section (meters).
    :param side_inner: Inner side length of the square cross-section (meters, for hollow rods).
    :return: Mass of the rod (kg).
    """
    density = 1.0  # Replace with the appropriate density (kg/m^3)
    mass = density * l * side ** 2
    hollow_mass = density * l * side_inner ** 2
    return mass - hollow_mass


def square_rod_inertia(l: float, side: float, side_inner: float = 0) -> pinocchio.pin.Inertia:
    """
    Calculates the inertia tensor of a solid or hollow square rod, assuming it is centered on the origin.
    
    :param l: Length of the rod (meters).
    :param side: Outer side length of the square cross-section (meters).
    :param side_inner: Inner side length of the square cross-section (meters, for hollow rods).
    :return: pinocchio.Inertia object representing the inertia of the rod.
    """
    mass = square_rod_mass(l, side, side_inner)
    lever = np.asarray([0, 0, 0])
    
    # Cross-section moment of inertia for a square (Ix and Iy)
    Ix = (1 / 12) * mass * (l ** 2 + (side ** 2 + side_inner ** 2) / 2)
    Iy = Ix  # Symmetric for x and y directions
    
    # Moment of inertia along the rod's axis (Iz)
    Iz = (1 / 6) * mass * ((side ** 2 + side_inner ** 2) / 2)
    
    # Construct the inertia matrix
    I = np.zeros((3, 3))
    I[0, 0] = Ix
    I[1, 1] = Iy
    I[2, 2] = Iz
    
    return pinocchio.pin.Inertia(mass, lever, I)

def create_inertia(inertial):
    """
    Creates a pinocchio.Inertia object from the provided data dictionary.
    
    :param data: Dictionary containing inertia, mass, and origin information.
    :return: pinocchio.Inertia object
    """
    # Step 1: Extract mass and convert to float
    mass = float(inertial['mass']['value'])
    
    # Step 2: Extract inertia components and build the 3x3 inertia matrix
    inertia_matrix = np.array([
        [float(inertial['inertia']['ixx']), float(inertial['inertia']['ixy']), float(inertial['inertia']['ixz'])],
        [float(inertial['inertia']['ixy']), float(inertial['inertia']['iyy']), float(inertial['inertia']['iyz'])],
        [float(inertial['inertia']['ixz']), float(inertial['inertia']['iyz']), float(inertial['inertia']['izz'])]
    ])
    
    # Step 3: Extract center of mass and convert to numpy array
    center_of_mass = np.array([float(coord) for coord in inertial['origin']['xyz'].split()])
    
    # Step 4: Create and return the Inertia object
    inertia = pinocchio.Inertia(mass, center_of_mass, inertia_matrix)
    return inertia

def body2connector_helper(joint_xyz, joint_rpy, link_xyz, link_rpy):
    """
    Computes the body2connector transformation based on joint and link origins.

    :param joint_xyz: [x, y, z] position of the joint relative to the parent link.
    :param joint_rpy: [roll, pitch, yaw] orientation of the joint in radians.
    :param link_xyz: [x, y, z] position of the link relative to its local frame.
    :param link_rpy: [roll, pitch, yaw] orientation of the link in radians.
    :return: Transformation object representing body2connector.
    """
    # Convert joint and link rpy to rotation matrices
    joint_rotation = Rotation.from_euler('xyz', joint_rpy).as_matrix()
    link_rotation = Rotation.from_euler('xyz', link_rpy).as_matrix()

    # Compute the relative translation
    relative_translation = np.array(joint_xyz) - np.array(link_xyz)

    # Compute the relative rotation
    relative_rotation = joint_rotation @ link_rotation.T


    return relative_rotation, relative_translation

def rpy_to_rotation_matrix(rpy):
    """
    Compute a rotation matrix from roll, pitch, and yaw angles (in radians).
    Args:
        roll (float): Rotation about the X-axis.
        pitch (float): Rotation about the Y-axis.
        yaw (float): Rotation about the Z-axis.
    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    Rx = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])
    
    # Rotation order: Rz (yaw) * Ry (pitch) * Rx (roll)
    return Rz @ Ry @ Rx

def urdf_to_dict(urdf_file_path):
    """
    Converts a URDF file into a Python dictionary.

    Args:
        urdf_file_path (str): Path to the URDF file.

    Returns:
        dict: A nested dictionary representing the URDF structure.
    """
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    
    def parse_element(element):
        """
        Recursively parses an XML element into a dictionary.

        Args:
            element (ET.Element): XML element to parse.

        Returns:
            dict: Dictionary representation of the XML element.
        """
        parsed = {key: element.attrib[key] for key in element.attrib}  # Attributes
        for child in element:
            child_tag = child.tag
            child_dict = parse_element(child)
            
            # Handle multiple elements with the same tag
            if child_tag not in parsed:
                parsed[child_tag] = child_dict
            else:
                if not isinstance(parsed[child_tag], list):
                    parsed[child_tag] = [parsed[child_tag]]
                parsed[child_tag].append(child_dict)

        return parsed

    # Parse the root element (robot)
    urdf_dict = {root.tag: parse_element(root)}
    return urdf_dict

# assets/Assem_4305_JOINT/Assem_4305_JOINT/urdf/Assem_4305_JOINT.urdf
# assets/Assem_4310_BASE/Assem_4310_BASE/urdf/Assem_4310_BASE.urdf
# assets/Assem_4310_JOINT/Assem_4310_JOINT/urdf/Assem_4310_JOINT.urdf
# Example usage
if __name__ == "__main__":
    #urdf_file = "assets/Assem_4305_JOINT/Assem_4305_JOINT/urdf/Assem_4305_JOINT.urdf"  # Replace with your file path
    #urdf_file = "assets/Assem_4310_JOINT/Assem_4310_JOINT/urdf/Assem_4310_JOINT.urdf"
    urdf_file = "assets/Assem_4310_BASE/Assem_4310_BASE/urdf/Assem_4310_BASE.urdf"
    urdf_dict = urdf_to_dict(urdf_file)
    pprint(urdf_dict)
