#! /usr/bin/env python

import roslib
import rospy
import rosbag
import math 
import random
import sys
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler

    
draw = rospy.Publisher('visualization_marker', Marker, queue_size=100)
bag = rosbag.Bag(sys.argv[1])

f = open(sys.argv[2], "w+")

cell_size = 20
cell_noise = cell_size/2.0
angle_discretize = 20
heading_noise = angle_discretize/2.0 
total_angle = 360       # For rotation
grid_length = 700       # In cm

# Define the sizes of the grid cells for movement
dim1 = grid_length/cell_size
dim2 = grid_length/cell_size
dim3 = total_angle/angle_discretize

#Deine the landmark positions in cm
tags = np.array([[125, 525], [125,325], [125,125], [425,125], [425, 325], [425,525]])

#Define the two probability arrays that will keep getting updated
belief = np.zeros((dim1, dim2, dim3))
belief_bar = np.zeros((dim1, dim2, dim3))

threshold = 0.1
belief[11, 27, int(total_angle/(2*angle_discretize))] = 1

lines = Marker()

# Add discrete noise to each movement
def addNoise(x, y, theta):
    x_mov = x*cell_size + cell_noise
    y_mov = y*cell_size + cell_noise
    rotation = theta*angle_discretize + heading_noise - 180
    return x_mov, y_mov, rotation
    

# visualize the landmarks as cubes on Rviz
def visualize_tags():
    global tags, draw
    for i in range(tags.shape[0]):

        grid_pos_x = tags[i,0]/100.0
        grid_pos_y = tags[i,1]/100.0
        landmarks = Marker()
        landmarks.header.frame_id = "landmarks"
        landmarks.header.stamp = rospy.Time.now()
        landmarks.ns = "tags"
        
        landmarks.id = i
        landmarks.type = Marker.CUBE
        landmarks.pose.position.x = grid_pos_x
        landmarks.pose.position.y = grid_pos_y
        landmarks.action = Marker.ADD
        landmarks.scale.x = 0.1
        landmarks.scale.y = 0.1
        landmarks.scale.z = 0.1 
        landmarks.color.g = 1.0
        landmarks.color.a = 1.0
        landmarks.lifetime = rospy.Duration()

        draw.publish(landmarks)

# Convert each angle to pi to -pi    
def normalize_angle_pi(theta_1, rot_2, rot_1):
    rot2 = theta_1 - rot_2
    rot1 = rot_1 - theta_1

    if rot1 < -(2*90):
        rot1 = rot1 + (2*180)
    if rot1 > (2*90):
        rot1 = rot1 - (2*180)

    if rot2 < -(2*90):
        rot2 = rot2 + (2*180)
    if rot2 > (2*90):
        rot2 = rot2 - (2*180)
    
    return rot2, rot1

# Observe where did the robot move in the grid
def findMovement_InGrid (x_c, x, y_c, y, theta_c, theta):
   
    x_mov1, y_mov1, rot_1 = addNoise(x, y, theta)
    x_mov2, y_mov2, rot_2 = addNoise(x_c, y_c, theta_c)

    x_diff = x_mov1 - x_mov2
    y_diff = y_mov1 - y_mov2

    total_trans = np.sqrt(x_diff**2 + y_diff**2)
    theta_1 = np.degrees(np.arctan2(y_diff, x_diff))

    rot2, rot1 = normalize_angle_pi(theta_1, rot_2, rot_1)
    return total_trans, rot2, rot1
    
# Calculate the probabilities of each position from a Gaussian distribution
def Gaussian(mean, variance, x):
    e = math.pow(np.e, -1.0*(((x-mean)**2)/(2.0*variance**2)))
    prob = e/(np.sqrt(2.0*np.pi)*variance)
    return prob

def writeToFile(output):
    global f
    f.write(output)
    f.write("\n")

# Display the moiton of the robot on Rviz
def display_lines (x, y):
    
    global belief, lines

    lines.header.frame_id = "/landmarks"
    lines.header.stamp = rospy.Time.now()
    lines.ns = "lines"
    lines.id = 0
    lines.type = Marker.LINE_STRIP
    p = Point()
    p.x = x
    p.y = y
    lines.points.append(p)
    lines.action = Marker.ADD
    lines.scale.x = 0.05

    lines.color.r = 1.0
    lines.color.a = 1.0
    draw.publish(lines)

# Find the absolute position of the robot in the 
def visualize_currentPos_InGrid(all_prob, drawLines):
    
    global belief
    belief = belief/all_prob
    
    dim3 = belief.shape[2]
    dim2 = belief.shape[1]
    dim1 = belief.shape[0]

    grid_index = np.argmax(belief)
    grid_index_angle = grid_index % dim3
    grid_index = grid_index / dim3

    grid_index_y = grid_index % dim2
    grid_index = grid_index / dim2
    grid_index_x = grid_index % dim1

    x, y, theta = addNoise(grid_index_x, grid_index_y, grid_index_angle)
    x = x/100.0
    y = y/100.0
    
    display_lines(x, y)

    #return +1 added values to the main function for the grid positions
    return grid_index_x+1, grid_index_y+1, grid_index_angle


#Calculate the probability of a robot from its current position to all other positions in the grid
def findTotalProb_InGrid(x_c, y_c, theta_c, rot1, rot2, trans):

    global belief, belief_bar

    all_prob = 0.0

    for x in range(belief.shape[0]):
        for y in range(belief.shape[1]):
            for theta in range(belief.shape[2]):
                trans_cmp, rot1_cmp, rot2_cmp = findMovement_InGrid(x_c, x, y_c, y, theta_c, theta) #Calculate the rotation, translation, rotation values from 
                #current position to every other position in the grid
                
                # Find probabilty of these rotation, translation, rotation from the motion model which is Gaussian in nature
                prob_rot1 = Gaussian(rot1, heading_noise, rot1_cmp)
                prob_rot2 = Gaussian(rot2, heading_noise, rot2_cmp)
                prob_trans = Gaussian(trans, cell_noise, trans_cmp)
                prob_rrt = prob_rot1 * prob_rot2 * prob_trans
                each_cell_prob = belief_bar[x_c, y_c, theta_c] * prob_rrt

                # Update the belief with the new probabilities
                belief[x,y,theta] = belief[x,y,theta] + each_cell_prob
                all_prob = all_prob + each_cell_prob

    return all_prob


def findPos_InGrid(rot1, rot2, trans):
    
    global belief, belief_bar, threshold

    belief_bar = belief
    belief = np.copy(belief_bar)
    all_prob = 0.0

    for x_c in range(belief.shape[0]):
        for y_c in range(belief.shape[1]):
            for theta_c in range(belief.shape[2]):
                
                if belief_bar[x_c, y_c, theta_c] < threshold:
                    continue
                
                else:
                    all_prob = findTotalProb_InGrid(x_c, y_c, theta_c, rot1, rot2, trans)

    #Find new current positions in the grid after motion                
    current_x, current_y, current_theta = visualize_currentPos_InGrid(all_prob, False)

    return current_x, current_y, current_theta

# Find required motion values to move to every other position in the grid from the current position
def comparePosition_otherCells(landmark, x, y, theta):
    global tags
    x_mov, y_mov, rot_mov = addNoise(x,y,theta)
    
    x_bel = tags[landmark,0] - x_mov
    y_bel = tags[landmark,1] - y_mov

    total_trans = ((-1.0)*x_bel)**2 + ((-1)*y_bel)**2
    total_trans = np.sqrt(total_trans)
    
    theta = np.arctan2(y_bel, x_bel)
    theta = np.degrees(theta)

    total_rot = theta - rot_mov

    if total_rot < -(2*90):
        total_rot = total_rot + (2*180)
    if total_rot > (2*90):
        total_rot = total_rot - (2*180)

    return total_trans, total_rot
    

def localize():

    global belief, belief_bar, bag, f

    writeToFile("Starting execution!\n\n")
    count = 1
    c1 = 1
    c2 = 2
    # bag = rosbag.Bag('grid.bag')

    t1, r1, r2 = findMovement_InGrid(12, 11, 28, 27, int((3*total_angle)/(4*angle_discretize)), int((total_angle)/(2*angle_discretize)))
    startingOutput = "Starting values " + str(r1) + "," + str(t1) + "," + str(r2)
    print startingOutput
    # writeToFile(startingOutput)

    for topic, msg, t in bag.read_messages():
        # visualize_tags()
        print "\n\n"
        print "Iteration", count
        if topic == "Movements":
            rot1, trans, rot2 = msg.rotation1, msg.translation, msg.rotation2
            rot1 = euler_from_quaternion([rot1.x, rot1.y, rot1.z, rot1.w])[2]
            rot1 = np.degrees(rot1)
            rot2 = euler_from_quaternion([rot2.x, rot2.y, rot2.z, rot2.w])[2]
            rot2 = np.degrees(rot2)

            x, y, heading = findPos_InGrid(rot1, rot2, trans)
            # visualize_tags()
            output = "AFTER MOVEMENT " + str(int((c1/2)+1)) + ", New current location of robot: [x, y, heading]- [" + str(x) + "," + str(y) + "," + str(heading) + "]"
            print output
            # writeToFile(output)
            c1+=2

        elif topic == "Observations":
            range_val = msg.range
            moved = range_val*100
            bearing_val = msg.bearing
            rot = euler_from_quaternion([bearing_val.x, bearing_val.y, bearing_val.z, bearing_val.w])[2]
            rot = np.degrees(rot)

            landmark = msg.tagNum
            belief_bar = belief
            belief = np.copy(belief_bar)

            all_prob = 0.0
            for x in range(belief.shape[0]):
                for y in range(belief.shape[1]):
                    for theta in range(belief.shape[2]):
                        comp_moved, comp_rot = comparePosition_otherCells(landmark, x, y, theta)
                        
                        # Calculate the probabilities of observing a particular position after having performed previous motion, from Sensor model
                        prob_moved = Gaussian(moved, cell_noise,comp_moved)
                        prob_rot = Gaussian(rot, heading_noise,comp_rot)
                        prob_mr = prob_moved*prob_rot
                        prb = belief_bar[x,y,theta]*prob_mr
                        belief[x,y,theta] = prb
                        all_prob = all_prob + prb
            
            x, y, heading = visualize_currentPos_InGrid(all_prob, True)

            output = "AFTER OBSERVATION " + str(int(c2/2)) + ", Observed location of robot: [x, y, heading]- [" + str(x) + "," + str(y) + "," + str(heading) + "]"
            print output
            writeToFile(output)
            c2+=2

        visualize_tags()
        count+=1    
    bag.close()
    f.close()
    print "\n\n"
    print "All execution successfully completed............................................ E X I T I N G !!!"
    exit()
    #rospy.spin()



if __name__ == '__main__':
    try:
        rospy.init_node('readBag', anonymous = True)
        details = "****************************************** CSE 568: Robotics Algorithms Fall 2018 ************************************************"
        writeToFile(details)
        details = "************************************************ Author: Aniruddha Sinha **********************************************************"
        writeToFile(details)
        details = "**************************************************** UBIT: asinha6 ****************************************************************"
        writeToFile(details)
        details = "************************************************* Person #: 50289428 **************************************************************"
        writeToFile(details)
        localize()
        # print "\n\n"
        # print "All execution successfully completed............................................ E X I T I N G !!!"
        # exit()
    except rospy.ROSInterruptException:
        pass
