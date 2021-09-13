BUILD:
mkdir build
cd build
cmake .. (CMakelists)
make (compile .cpp)

RUN:
roscore
GAZEBO_PLUGIN_PATH=. gazebo --verbose ../../set_velocity.world

