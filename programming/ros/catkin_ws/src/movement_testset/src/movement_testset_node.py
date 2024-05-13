#!/usr/bin/env python3
"""
    philippvincent.ebert@stud.tu-darmstadt.de
"""

import os

import movement_generator as mg
import numpy as np
import pandas as pd
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

initialized = False
joint_state_time_stamp = None
joint_state_position_joint1 = None
joint_state_velocity_joint1 = None
joint_state_effort_joint1 = None
joint_state_position_joint2 = None
joint_state_velocity_joint2 = None
joint_state_effort_joint2 = None
joint_state_position_joint3 = None
joint_state_velocity_joint3 = None
joint_state_effort_joint3 = None
joint_state_position_joint4 = None
joint_state_velocity_joint4 = None
joint_state_effort_joint4 = None

def make_traj_msg(n_points, poss, vels, times):
    traj_msg = JointTrajectory()
    now = rospy.Time.now()
    traj_msg.header.stamp = now
    traj_msg.joint_names = [
        "joint_1_from_motor_enc",
        "joint_2_from_motor_enc",
        "joint_3_from_motor_enc",
        "joint_4_from_motor_enc"
    ]

    if n_points == 1:
        traj_msg.points = [
            JointTrajectoryPoint(
                positions=poss.flatten(),
                velocities=vels.flatten(),
                time_from_start=rospy.Duration(times),
            )
        ]
    else:
        traj_msg.points = [
            JointTrajectoryPoint(
                positions=poss[i],
                velocities=vels[i],
                time_from_start=rospy.Duration(times[i]),
            )
            for i in range(n_points)
        ]
    return traj_msg


def calculate_acceleration(velocities):
    gaussian_filter = np.array([1, 6, 15, 20, 15, 6, 1]) / 64

    # for i in range(velocities):
    q_smooth = np.convolve(velocities, gaussian_filter, mode="same")

    dq = np.gradient(q_smooth)

    # set first and last values to zero to avoid edge effects
    n_off = 4
    dq[:n_off] = 0
    dq[-n_off:] = 0
    return dq


def callback(msg):
    global initialized
    global joint_state_position_joint1
    global joint_state_velocity_joint1
    global joint_state_effort_joint1
    global joint_state_position_joint2
    global joint_state_velocity_joint2
    global joint_state_effort_joint2
    global joint_state_position_joint3
    global joint_state_velocity_joint3
    global joint_state_effort_joint3
    global joint_state_position_joint4
    global joint_state_velocity_joint4
    global joint_state_effort_joint4
    if initialized:
        joint_state_position_joint1 = np.append(
            joint_state_position_joint1, msg.position[1]
        )
        joint_state_velocity_joint1 = np.append(
            joint_state_velocity_joint1, msg.velocity[1]
        )
        joint_state_effort_joint1 = np.append(joint_state_effort_joint1, msg.effort[1])
        joint_state_position_joint2 = np.append(
            joint_state_position_joint2, msg.position[3]
        )
        joint_state_velocity_joint2 = np.append(
            joint_state_velocity_joint2, msg.velocity[3]
        )
        joint_state_effort_joint2 = np.append(joint_state_effort_joint2, msg.effort[3])
        joint_state_position_joint3 = np.append(
            joint_state_position_joint3, msg.position[5]
        )
        joint_state_velocity_joint3 = np.append(
            joint_state_velocity_joint3, msg.velocity[5]
        )
        joint_state_effort_joint3 = np.append(joint_state_effort_joint3, msg.effort[5])
        joint_state_position_joint4 = np.append(
            joint_state_position_joint4, msg.position[7]
        )
        joint_state_velocity_joint4 = np.append(
            joint_state_velocity_joint4, msg.velocity[7]
        )
        joint_state_effort_joint4 = np.append(joint_state_effort_joint4, msg.effort[7])
    else:
        joint_state_position_joint1 = np.array(msg.position[1])
        joint_state_velocity_joint1 = np.array(msg.velocity[1])
        joint_state_effort_joint1 = np.array(msg.effort[1])
        joint_state_position_joint2 = np.array(msg.position[3])
        joint_state_velocity_joint2 = np.array(msg.velocity[3])
        joint_state_effort_joint2 = np.array(msg.effort[3])
        joint_state_position_joint3 = np.array(msg.position[5])
        joint_state_velocity_joint3 = np.array(msg.velocity[5])
        joint_state_effort_joint3 = np.array(msg.effort[5])
        joint_state_position_joint4 = np.array(msg.position[7])
        joint_state_velocity_joint4 = np.array(msg.velocity[6])
        joint_state_effort_joint4 = np.array(msg.effort[7])
        initialized = True


def main():
    rospy.init_node("movement_testset_node", anonymous=True)
    # rospy.Rate(500)  # 500hz

    rospy.loginfo("Connecting to robot...")
    # subscribe to joint states
    joint_state_sub = rospy.Subscriber(
        "/wam/joint_states", JointState, callback
    )
    # wait Subscriber to start
    rospy.wait_for_message("/wam/joint_states", JointState)

    # publish test movements
    test_movements_pub = rospy.Publisher(
        "/wam/motor_enc_joint_trajectory_controller/command",
        JointTrajectory,
        queue_size=10,
    )

    # wait Publisher to start
    rospy.sleep(1)

    print("\n    Ready!\n")

    # while not rospy.is_shutdown():
    # initialize movement generator for all four joints
    mg_joint1 = mg.MovementGenerator(duration=10 * 1)
    mg_joint2 = mg.MovementGenerator(duration=10 * 1)
    mg_joint3 = mg.MovementGenerator(duration=10 * 1)
    mg_joint4 = mg.MovementGenerator(duration=10 * 1)
    movements_joint1 = mg_joint1.generate()
    movements_joint2 = mg_joint2.generate()
    movements_joint3 = mg_joint3.generate()
    movements_joint4 = mg_joint4.generate()
    movements_joint1 = np.reshape(movements_joint1, (-1, 1))
    movements_joint2 = np.reshape(movements_joint2, (-1, 1))
    movements_joint3 = np.reshape(movements_joint3, (-1, 1))
    movements_joint4 = np.reshape(movements_joint4, (-1, 1))
    poss = np.zeros((len(movements_joint1), 3))
    poss = np.insert(poss, [0], movements_joint1, axis=1)
    poss = np.insert(poss, [1], movements_joint2, axis=1)
    poss = np.insert(poss, [2], movements_joint3, axis=1)
    poss = np.insert(poss, [3], movements_joint4, axis=1)
    vels = np.zeros_like(poss)
    times = np.linspace(0, 10 * 1, len(poss))

    # make sure robot is in start position, 0.2 randians tolerance due to low gains
    # make sure you have a smooth transition from one to the next set of movements

    test_movements_pub.publish(make_traj_msg(len(times), poss, vels, times))
    rospy.sleep(10)

    dataset = pd.DataFrame(
        [
            joint_state_position_joint1,
            joint_state_velocity_joint1,
            calculate_acceleration(joint_state_velocity_joint1),
            joint_state_effort_joint1,
            joint_state_position_joint2,
            joint_state_velocity_joint2,
            calculate_acceleration(joint_state_velocity_joint2),
            joint_state_effort_joint2,
            joint_state_position_joint3,
            joint_state_velocity_joint3,
            calculate_acceleration(joint_state_velocity_joint3),
            joint_state_effort_joint3,
            joint_state_position_joint4,
            joint_state_velocity_joint4,
            calculate_acceleration(joint_state_velocity_joint4),
            joint_state_effort_joint4,
        ]
    ).T

    dataset = dataset.rename(
        columns={
            0: "joint_state_pos_joint1",
            1: "joint_state_vel_joint1",
            2: "joint_state_acc_joint1",
            3: "joint_state_effort_joint1",
            4: "joint_state_pos_joint2",
            5: "joint_state_vel_joint2",
            6: "joint_state_acc_joint2",
            7: "joint_state_effort_joint2",
            8: "joint_state_pos_joint3",
            9: "joint_state_vel_joint3",
            10: "joint_state_acc_joint3",
            11: "joint_state_effort_joint3",
            12: "joint_state_pos_joint4",
            13: "joint_state_vel_joint4",
            14: "joint_state_acc_joint4",
            15: "joint_state_effort_joint4",
        }
    )

    print(os.getcwd())
    dataset.to_csv("testset.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
