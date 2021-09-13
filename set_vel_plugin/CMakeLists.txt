
cmake_minimum_required(VERSION 2.6)
project(SetVelPlugin)

find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${geometry_msgs_INCLUDE_DIRS})

find_package(gazebo REQUIRED)

include_directories(include ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(SetLinkVelocityPlugin SHARED src/SetLinkVelocityPlugin.cpp)
target_link_libraries(SetLinkVelocityPlugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})

