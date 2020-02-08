# ROS_GridLocalization_BayesFilter_CSE568
Use discrete Bayes Localization to implement grid localization on robots in ROS.
<br>Grid Localization is a variant of discrete Bayes Localization. In this method, the map
is an occupancy grid. At each time step, the algorithm finds out the probabilities of
the robot presence at every grid cell. The grid cells with maximum probabilities at each
step, characterize the robot's trajectory. Grid Localization runs in two iterative steps - <b>Movement and Observation</b>.
After each movement, we compute if that movement can move the robot between
grid cells. For each observation, find the most probable cells where that
measurement could have occurred.

<br><br><b>Input Data - </b>
<br>Rosbag is used for dumping the messages published on certain topics to files. We have a bag file which includes the movements and
observation information from one scenario. Read the bag file to simulate
the localization algorithm running on a robot executing this scenario.

<br><br><b>Visualization in RViz-</b>
<br>After running the grid localization algorithm, we draw the final trajectory as a marker line
in Rviz.
