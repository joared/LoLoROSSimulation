/*
 * Copyright (C) 2017 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/PhysicsIface.hh>
#include <gazebo/physics/PhysicsEngine.hh>

#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"
#include "geometry_msgs/Twist.h"

namespace gazebo
{
  //////////////////////////////////////////////////
  /// \brief Sets velocity on a link or joint
  class SetLinkVelocityPlugin : public ModelPlugin
  {
    /// \brief a pointer to the model this plugin was loaded by
    public: physics::ModelPtr model;
    /// \brief object for callback connection
    public: event::ConnectionPtr updateConnection;
    /// \brief number of updates received
    public: int update_num = 0;

    /// \brief A node use for ROS transport
    private: std::unique_ptr<ros::NodeHandle> rosNode;

    /// \brief A ROS subscriber
    private: ros::Subscriber rosSub;
    private: ros::Subscriber rosPub;

    /// \brief A ROS callbackqueue that helps process messages
    private: ros::CallbackQueue rosQueue;

    /// \brief A thread the keeps running the rosQueue
    private: std::thread rosQueueThread;

    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
      {
        this->model = _model;
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&SetLinkVelocityPlugin::Update, this, std::placeholders::_1));

	if (!ros::isInitialized())
	{
	  int argc = 0;
	  char **argv = NULL;
	  ros::init(argc, argv, "gazebo_client",
	      ros::init_options::NoSigintHandler);
	}

	// Create our ROS node. This acts in a similar manner to
	// the Gazebo node
	this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

	// Create a named topic, and subscribe to it.
	ros::SubscribeOptions so =
	  ros::SubscribeOptions::create<geometry_msgs::Twist>(
	      "/" + this->model->GetName() + "/vel_cmd",
	      1,
	      boost::bind(&SetLinkVelocityPlugin::OnRosMsg, this, _1),
	      ros::VoidPtr(), &this->rosQueue);
	this->rosSub = this->rosNode->subscribe(so);

	//this->rosPub = this->rosNode->advertise<geometry_msgs::Pose>("chatter", 1000);

	// Spin up the queue helper thread.
	this->rosQueueThread =
	  std::thread(std::bind(&SetLinkVelocityPlugin::QueueThread, this));

      }

    public: void Update(const common::UpdateInfo &_info)
      {
        if (update_num == 0)
        {
          // Link velocity instantaneously without applying forces
          //model->GetLink("white_link_0")->SetLinearVel({0, 1, 0});
          //model->GetLink("white_link_1")->SetLinearVel({0, 1, 0});
          //model->GetLink("white_link_1")->SetAngularVel({1, 0, 0});
          model->GetLink("link_0")->SetAngularVel({1, 0, 0});
	  printf("Update num: %d", update_num);
     }
        else if (update_num == 10000)
        {
          //model->GetLink("white_link_0")->SetLinearVel({0, 0, 0});
          //model->GetLink("white_link_1")->SetLinearVel({0, 0, 0});
          //model->GetLink("white_link_1")->SetAngularVel({0, 0, 0});
          model->GetLink("link_0")->SetAngularVel({0, 0, 0});
        }
        update_num++;
      }

    public: void SetTwist() {}

    /// \brief Handle an incoming message from ROS
    /// \param[in] _msg A float value that is used to set the velocity
    /// of the Velodyne.
    public: void OnRosMsg(const geometry_msgs::TwistConstPtr &_msg)
    {
      //this->SetVelocity(_msg->data);
      //model->GetLink("white_link_2")->SetLinearVel({_msg->linear.x, _msg->linear.y, _msg->linear.z});
      //model->GetLink("white_link_2")->SetAngularVel({_msg->angular.x, _msg->angular.y, _msg->angular.z});
      //model->GetJoint("link_0_JOINT_1")->SetVelocity(0, 1);
    }

    /// \brief ROS helper function that processes messages
    private: void QueueThread()
    {
      static const double timeout = 0.01;
      while (this->rosNode->ok())
      {
        this->rosQueue.callAvailable(ros::WallDuration(timeout));
      }
    }
  };

  GZ_REGISTER_MODEL_PLUGIN(SetLinkVelocityPlugin)
}
