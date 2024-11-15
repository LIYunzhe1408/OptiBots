## What this file do
Based on the predefined URDF, implement converting scratch code with urdfpy and animate it to validate.


| URDF Name         | Visualize              |
| :-----------------| :--------------------  | 
| Assem_4310_BASE   | ![](./visualization/Assem_4310_BASE.png) |
| Assem_4310_JOINT  | ![](./visualization/Assem_4310_JOINT.png)|
| Assem_4305_JOINT  | ![](./visualization/Assem_4305_JOINT.png)|

## How to start
1. Set up with the `requirement.txt`
   1. cd `/TimorExamples/stl2urdf/`
   2. run `pip install -r requirements.txt`
2. Take a look in `/generatedURDF/convert.py`. Scroll down to the bottom, set `usePredefinedURDF` to True or False.
    * For reachability search, use predefined URDF is recommended as assembling code need further working.

## Next step
Generalize converting file with clear user input as command argv to adjust parameters.

