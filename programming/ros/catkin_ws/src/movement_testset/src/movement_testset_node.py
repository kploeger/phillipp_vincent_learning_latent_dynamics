#!/usr/bin/env python3
"""
    philippvincent.ebert@stud.tu-darmstadt.de
"""

import os

import dynamic_reconfigure.client
import movement_generator as mg
import numpy as np
import pandas as pd
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

joint_states = False
joint_state_time_stamp = None
joint_state_position_joint4 = None
joint_state_velocity_joint4 = None
joint_state_effort_joint4 = None
joint_state_position_joint3 = None
joint_state_velocity_joint3 = None
joint_state_effort_joint3 = None
joint_state_position_joint2 = None
joint_state_velocity_joint2 = None
joint_state_effort_joint2 = None
joint_state_position_joint1 = None
joint_state_velocity_joint1 = None
joint_state_effort_joint1 = None


def set_jan_gains():
    gains = [
        {"p": 200, "d": 7, "i": 0},
        {"p": 300, "d": 15, "i": 0},
        {"p": 100, "d": 5, "i": 0},
        {"p": 100, "d": 2.5, "i": 0},
    ]
    for i in range(4):
        client = dynamic_reconfigure.client.Client(
            "/wam4/joint_effort_trajectory_controller/gains/wam4_joint_" + str(i + 1)
        )
        client.update_configuration(gains[i])


def make_traj_msg(n_points, poss, vels, times):
    traj_msg = JointTrajectory()
    now = rospy.Time.now()
    traj_msg.header.stamp = now
    traj_msg.joint_names = [
        "wam4_joint_1",
        "wam4_joint_2",
        "wam4_joint_3",
        "wam4_joint_4",
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


def callback(data):
    global joint_states
    global joint_state_position_joint4
    global joint_state_velocity_joint4
    global joint_state_effort_joint4
    global joint_state_position_joint3
    global joint_state_velocity_joint3
    global joint_state_effort_joint3
    global joint_state_position_joint2
    global joint_state_velocity_joint2
    global joint_state_effort_joint2
    global joint_state_position_joint1
    global joint_state_velocity_joint1
    global joint_state_effort_joint1
    if joint_states:
        joint_state_position_joint4 = np.append(
            joint_state_position_joint4, data.position[3]
        )
        joint_state_velocity_joint4 = np.append(
            joint_state_velocity_joint4, data.velocity[3]
        )
        joint_state_effort_joint4 = np.append(joint_state_effort_joint4, data.effort[3])
        joint_state_position_joint3 = np.append(
            joint_state_position_joint3, data.position[2]
        )
        joint_state_velocity_joint3 = np.append(
            joint_state_velocity_joint3, data.velocity[2]
        )
        joint_state_effort_joint3 = np.append(joint_state_effort_joint3, data.effort[2])
        joint_state_position_joint2 = np.append(
            joint_state_position_joint2, data.position[1]
        )
        joint_state_velocity_joint2 = np.append(
            joint_state_velocity_joint2, data.velocity[1]
        )
        joint_state_effort_joint2 = np.append(joint_state_effort_joint2, data.effort[1])
        joint_state_position_joint1 = np.append(
            joint_state_position_joint1, data.position[0]
        )
        joint_state_velocity_joint1 = np.append(
            joint_state_velocity_joint1, data.velocity[0]
        )
        joint_state_effort_joint1 = np.append(joint_state_effort_joint1, data.effort[0])
    else:
        joint_state_position_joint4 = np.array(data.position[3])
        joint_state_velocity_joint4 = np.array(data.velocity[3])
        joint_state_effort_joint4 = np.array(data.effort[3])
        joint_state_position_joint3 = np.array(data.position[2])
        joint_state_velocity_joint3 = np.array(data.velocity[2])
        joint_state_effort_joint3 = np.array(data.effort[2])
        joint_state_position_joint2 = np.array(data.position[1])
        joint_state_velocity_joint2 = np.array(data.velocity[1])
        joint_state_effort_joint2 = np.array(data.effort[1])
        joint_state_position_joint1 = np.array(data.position[0])
        joint_state_velocity_joint1 = np.array(data.velocity[0])
        joint_state_effort_joint1 = np.array(data.effort[0])
        joint_states = True


def main():
    rospy.init_node("movement_testset_node", anonymous=True)
    # rospy.Rate(500)  # 500hz

    # publish test movements
    test_movements_pub = rospy.Publisher(
        # "/wam4/joint_effort_trajectory_controller/test_movements",
        "/wam4/joint_effort_trajectory_controller/command",
        JointTrajectory,
        queue_size=10,
    )

    # wait Publisher to start
    rospy.sleep(1)

    # change gains to the ones Jan came up with
    set_jan_gains()

    print("\n    Ready!\n")

    # while not rospy.is_shutdown():
    movement_gen = mg.MovementGenerator(duration=10 * 1)
    movements = movement_gen.generate()
    movements = np.reshape(movements, (-1, 1))
    poss = np.zeros((len(movements), 3))
    poss = np.insert(poss, [3], movements, axis=1)
    vels = np.zeros_like(poss)
    times = np.linspace(0, 10 * 1, len(poss))

    # make sure robot is in start position, 0.2 randians tolerance due to low gains
    # make sure you have a smooth transition from one to the next set of movements

    # subscribe to joint states
    joint_states_sub = rospy.Subscriber("/wam4/joint_states", JointState, callback)

    # wait Subscriber to start
    # rospy.sleep(1)

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
