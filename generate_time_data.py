import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time
from tqdm import tqdm
import copy


def generate(i):
    # Instantiate Robotarium object
    N = 50

    x = np.random.uniform(-1.5, 1.5, N)
    y = np.random.uniform(-0.9, 0.9, N)
    angle = np.random.uniform(0, 2 * np.pi, N)
    initial_pose = np.stack((x, y, angle))

    x_goal = np.random.uniform(-1.5, 1.5, N)
    y_goal = np.random.uniform(-0.9, 0.9, N)
    angle_goal = np.zeros(N)
    goal_points = np.stack((x_goal, y_goal, angle_goal))
    goal_position = np.stack((x_goal, y_goal))

    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_pose, sim_in_real_time=False)

    # Define goal points by removing orientation from poses
    # goal_points = generate_initial_conditions(N)

    # Create unicycle position controller
    unicycle_position_controller = create_clf_unicycle_position_controller()

    # Create barrier certificates to avoid collision
    uni_barrier_cert = create_unicycle_barrier_certificate()

    # define x initially
    x = r.get_poses()
    prev_pos = x.T.copy()
    r.step()

    count_step = np.zeros(N)
    dist = np.zeros(N)
    # While the number of robots at the required poses is less
    # than N...
    arrived = at_pose(x, goal_points, rotation_error=100, position_error=0.1)
    while (np.size(arrived) != N):

        # Get poses of agents
        x = r.get_poses()
        pos = x.T
        dist += np.linalg.norm(pos[:, :2] - prev_pos[:, :2], axis=1)
        if arrived:
            count_step[arrived] -= 1

        # Create single-integrator control inputs
        dxu = unicycle_position_controller(x, goal_points[:2][:])

        # Create safe control inputs (i.e., no collisions)
        # dxu = uni_barrier_cert(dxu, x)
        # print(dxu.shape)

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Iterate the simulation
        prev_pos = x.T.copy()
        r.step()
        arrived = at_pose(x, goal_points, rotation_error=100)
        count_step += 1
        print(np.sum(dist))

    # Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()

    np.save(f'./data/initial_pose/initial_pose_{i}.npy', initial_pose)
    np.save(f'./data/goal_position/goal_position_{i}.npy', goal_position)
    np.save(f'./data/iteration/iteration_{i}.npy', count_step)

    del r


for k in tqdm(range(500)):
    generate(k)
