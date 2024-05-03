#!/usr/bin/env python3
"""
    philippvincent.ebert@stud.tu-darmstadt.de
"""

import dynamic_reconfigure.client
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# all positions are transformed to the message format, we only use the positions of joints_from_motor_enc

poss_start1 = np.array(
    [
        [0, 0, 0, 0],
    ]
)
vels_start1 = np.zeros_like(poss_start1)
times_start1 = np.array([0])

poss_start2 = np.array(
    [
        [0, 0.5, 1.7, 0.1],
    ]
)
vels_start2 = np.zeros_like(poss_start2)
times_start2 = np.array([0])

poss_hello1 = np.array(
    [
        [0, -0.7, 0, 0.1],
        [0, 0.5, 0, 1.2],
        [0, -0.5, 0, 0.1],
        [0, 0.5, 0, 1.2],
    ]
)
vels_hello1 = np.zeros_like(poss_hello1)
times_hello1 = np.array([0.5, 1, 1.5, 2])

poss_hello2 = np.array(
    [
        [0, 0.5, 1.7, 1.8],
        [0, 0.5, 1.7, 1],
        [0, 0.5, 1.7, 1.8],
        [0, 0.5, 1.7, 1],
    ]
)
vels_hello2 = np.zeros_like(poss_hello2)
times_hello2 = np.array([0.3, 0.6, 0.9, 1.2])

def make_traj_msg(n_points, poss, vels, times):
    traj_msg = JointTrajectory()
    now = rospy.Time.now()
    traj_msg.header.stamp = now
    traj_msg.joint_names = [
        "joint_1_from_motor_enc",
        "joint_2_from_motor_enc",
        "joint_3_from_motor_enc",
        "joint_4_from_motor_enc",
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


class Menu:
    def __init__(self, default_duration=10.0):
        self.duration = default_duration
        self.greeting = 1
        self.done = False

    def __call__(self):
        while True:
            menu_input = input("Choose: again/set_duration/greeting/quit? (a/d/g/q): ")
            if menu_input in ["", "a", "A", "again", "Again"]:
                break
            elif menu_input in ["d", "D", "duration", "change duration"]:
                duration_input = input("How many seconds to juggle for?\n")
                try:
                    dur = float(duration_input)
                    if dur < 1.0:
                        print("Wink duration has to be longer than 1s.")
                    else:
                        self.duration = dur
                        print(
                            "Wink duration changed to {}s for all future runs.".format(
                                self.duration
                            )
                        )
                except ValueError:
                    print("Huh?")
            elif menu_input in ["g", "G"]:
                greeting_input = input(
                    "How do you want to be greeted, enthusiastically or gentle? (1/2)\n"
                )
                self.greeting = int(greeting_input)
            elif menu_input in ["q", "Q", "quit", "Quit"]:
                self.done = True
                break
            else:
                print("Huh?")


def main():
    rospy.init_node("hello_world_node", anonymous=True)

    instructions = """
    ----------------------------------
    Hello World for the WAM4 robot:
    ----------------------------------

    To run the demo, you need to start 4 components. 

    1. The roscore:
        roscore
    2. The robot controller:
        roslaunch wam_bringup wam.launch controller:=joint_effort_trajectory_controller
    3. This hello world node;
        roslaunch hello_world hello_world.launch

    Order matters! Ctrl-C if you launched this demo before roscore.
    """
    print(instructions)

    pos = np.zeros(4)

    def joint_pos_callback(msg):
        # The TrajectoryMessage we get back has the array sorted in a different way (by the names)
        # So this is how we get the array from trajectory.joint_names
        pos[:] = [msg.position[1], msg.position[3], msg.position[5], msg.position[7]]

    rospy.loginfo("Connecting to robot...")
    joint_state_sub = rospy.Subscriber(
        "/wam/joint_states", JointState, joint_pos_callback
    )
    rospy.wait_for_message("/wam/joint_states", JointState)
    command_pub = rospy.Publisher(
        "/wam/motor_enc_joint_trajectory_controller/command",
        JointTrajectory,
        queue_size=10,
    )

    # wait publisher to start
    rospy.sleep(1)

    print("\n    Ready to say hello!\n")

    menu = Menu(default_duration=10.0)
    menu()

    while not menu.done:
        # move to start position
        move_duration = 5
        t0 = rospy.Time.now()
        if menu.greeting == 1:
            command_pub.publish(
                make_traj_msg(1, poss_start1[0], vels_start1[0], move_duration)
            )
        elif menu.greeting == 2:
            command_pub.publish(
                make_traj_msg(1, poss_start2[0], vels_start2[0], move_duration)
            )

        time_taken = rospy.Time.now() - t0

        time_to_sleep = max(0, move_duration - time_taken.to_sec())
        rospy.sleep(time_to_sleep)

        # make sure robot is in start position, 0.2 randians tolerance due to low gains
        if menu.greeting == 1:
            print(pos)
            print(poss_start1[0])
            assert np.allclose(
                pos, poss_start1[0], atol=0.2
            ), "Robot is not in start position!"
        elif menu.greeting == 2:
            print(pos)
            print(poss_start2[0])
            assert np.allclose(
                pos, poss_start2[0], atol=0.2
            ), "Robot is not in start position!"

        # start wink
        input("Press enter to start wink!")
        t0 = rospy.Time.now()

        # wait a bit for correct timing to catch the ball
        rospy.sleep(0.25)

        # loop cyclic MP
        while t0 + rospy.Duration(menu.duration) > rospy.Time.now():
            # while True:
            if menu.greeting == 1:
                command_pub.publish(
                    make_traj_msg(
                        len(times_hello1), poss_hello1, vels_hello1, times_hello1
                    )
                )
                rospy.sleep(times_hello1[-1])
            elif menu.greeting == 2:
                command_pub.publish(
                    make_traj_msg(
                        len(times_hello2), poss_hello2, vels_hello2, times_hello2
                    )
                )
                rospy.sleep(times_hello2[-1])

        # menu
        menu()


if __name__ == "__main__":
    main()
