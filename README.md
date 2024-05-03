For the python scripts to work you need a virtual python environment:

```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install matplotlib scipy tensorflow seaborn pandas
```

The newest version of python (3.12) currently does not support tensorflow.

To build the catkin workspace correctly navigate to the wam_core submodule of this repository
and read the README.md. You need the catkin_tools. They do not come with ros noetic.

For RViz to work you must set the frame to world and add a RobotModel. Then the 3D model should appear.

Every catkin build autogenerates only from the first .srv file each run. So the build fails 2 times before it works.

```bash
    roslaunch hello_world hello_world.launch
```
to explore an example program.
You find it in catkin_ws/src/hello_world/src/hello_world_node.py.

The wam_examples do not work. They use the deprecated wam4/7.launch files from wam_bringup.
