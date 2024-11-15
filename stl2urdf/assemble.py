from urdfpy import URDF

"""
Run HERE
"""
usePredefined = False

# robot = URDF.load('../assets/Assem_4310_BASE/Assem_4310_BASE/urdf/test.urdf')
robot = URDF.load('./master.urdf')
for link in robot.links:
    print(link.name)
robot.show()
