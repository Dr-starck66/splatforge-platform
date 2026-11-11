#!/bin/bash
set -u
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_HOSTNAME=127.0.0.1
mkdir -p /tmp/onorbit
roscore >/tmp/onorbit/roscore.log 2>&1 & echo $! >/tmp/onorbit/roscore.pid
sleep 4
# Headless Gazebo is a real /clock source, not a mocked clock message.
roslaunch gazebo_ros empty_world.launch gui:=false paused:=false headless:=true >/tmp/onorbit/gazebo.log 2>&1 & echo $! >/tmp/onorbit/gazebo.pid
sleep 5
# Real upstream C++ code dynamically evaluates orbital state from /clock.
roslaunch orbit_publisher_pkg basic.launch >/tmp/onorbit/orbit.log 2>&1 & echo $! >/tmp/onorbit/orbit.pid
exec python3 -u /opt/onorbitros-public-probe/api.py
