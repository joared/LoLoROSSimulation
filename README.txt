BUILD:
mkdir build
cd build
cmake .. (CMakelists)
make (compile .cpp)

RUN:
roscore
GAZEBO_PLUGIN_PATH=. gazebo --verbose ../../set_velocity.world

CAMERA CALIBRATION:

uvcdynctrl --device=/dev/video0 --clist
uvcdynctrl --device=/dev/video0 --set='Focus, Auto' 0
uvcdynctrl --device=/dev/video0 --get='Focus, Auto'

(rosrun usb_cam usb_cam_node)
instead:
roslaunch feature_detection usb_cam-test.launch
rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.021 image:=/usb_cam/image_raw camera:=/usb_cam --no-service-check

ROS_NAMESPACE=usb_cam rosrun image_proc image_proc
