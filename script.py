import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from enum import Enum
from tqdm import tqdm


class Task(Enum):
    IDLE = 0
    GO_TO_PICK = 1
    GO_TO_COLLECTION = 2
    GO_TO_RECHARGE = 3
    RECHARGING = 4
    WAIT_FOR_RECHARGING = 5
    DEAD = 6


# Instantiate Robotarium object
N = 5
# initialization_loc = np.random.uniform(low=-2, high=-2, size=(3, N))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# TODO locations
# Task locations
task_loc_list = np.array([[0, 0], [0, 0.8], [0, -0.8], [0.8, 0], [-0.8, 0]])
task_loc2ind = {(0, 0): 0,
                (0, 0.8): 1,
                (0, -0.8): 2,
                (0.8, 0): 3,
                (-0.8, 0): 4}
task_ind2loc = {task_loc2ind[i]: i for i in task_loc2ind}

# recharging location
recharging_loc = np.array([[-1.3, 0.8], [1.3, 0.8]])
recharging_loc2ind = {(-1.3, 0.8): 0,
                      (1.3, 0.8): 1}
recharging_ind2loc = {recharging_loc2ind[i]: i for i in recharging_loc2ind}

# status of recharging location
recharging_available = np.array([True, True])

# collect bin location
collect_loc = np.array([-1.3, -0.8])

# distance clearance threshold
distance_threshold = 0.1

# Number of treasure collected
no_task_completed = 0

# status of Treasure
is_treasure_picked_up = np.array([False] * N)

# battery level of robots
battery_level = np.array([100.0] * N)

# distance traveled since last iteration
dist_travel = np.array([0.0] * N)

# status of robots
is_alive = np.array([True] * N)

# status of recharging
is_recharging = np.array([False] * N)

# status of moving
is_moving = np.array([False] * N)

# status of availability
is_task_available = np.array([True] * N)

# status of task
is_assigned = np.array([False] * N)

# available robots indice
available_robots_indice = list(range(N))

# current task
current_task_list = [Task.IDLE] * N
current_goal_loc = np.zeros((N, 2))
current_treasure_list = [None] * 5

# How many iterations do we want (about N*0.033 seconds)
iterations = 10000


def energy_loss(prev_energy, dist, is_treasure, alpha=0.01, beta=1, gamma=0.1):  # TODO alpha, beta
    return prev_energy - (alpha + beta * dist + gamma * is_treasure)


def energy_gain(prev_energy, delta=0.5):
    return prev_energy + delta


def task_assign(task_loc, robot_loc):
    """
    :param task_loc: locations of tasks, [[g1_x, g1_y], [g2_x, g2_y], ...]
    :param robot_loc: locations of robots, [[r1_x, r1_y], [r2_x, r2_y], ...]
    :return: the task loc corresponding to the robot_loc
    """
    assigned_task_loc = np.array([123, 123])
    for i in range(robot_loc.shape[0]):
        distances = np.linalg.norm(robot_loc[i] - task_loc, axis=1)
        sorted_ind = np.argsort(distances)
        for j in sorted_ind:
            task_index = task_loc2ind[tuple(task_loc[j])]
            if is_task_available[task_index]:
                assigned_task_loc = np.vstack((assigned_task_loc, task_loc[j]))
                break
    return assigned_task_loc[1:]


# We're working in single-integrator dynamics, and we don't want the robots
# to collide or drive off the testbed.  Thus, we're going to use barrier certificates
# si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary()

# Create SI to UNI dynamics tranformation
# si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()

prev_x = r.get_poses().T
goals = prev_x[:, :2]
prev_battery = battery_level
r.step()

for k in tqdm(range(iterations)):
    x = r.get_poses().T
    dist_travel = np.linalg.norm(x[:, :2] - prev_x[:, :2], axis=1)
    for i in range(N):
        current_robot_task = current_task_list[i]

        # robot is dead?
        if battery_level[i] < 1:
            current_task_list[i] = Task.DEAD
            is_alive[i] = False
        if current_robot_task == Task.DEAD:
            continue

        # robot is waiting for recharging?
        if current_robot_task == Task.WAIT_FOR_RECHARGING:
            available_index = np.where(recharging_available)[0]
            if np.size(available_index) == 0:
                continue
            current_goal_loc[i] = recharging_loc[available_index[0]]
            recharging_available[available_index[0]] = False
            current_task_list[i] = Task.GO_TO_RECHARGE

        # robot is going to pick up?
        if current_robot_task == Task.GO_TO_PICK:
            # the robot already reaches the treasure
            if np.linalg.norm(x[i, :2] - current_goal_loc[i]) < distance_threshold:
                is_treasure_picked_up[i] = True
                current_goal_loc[i] = collect_loc
                current_task_list[i] = Task.GO_TO_COLLECTION

        # robot is going to the collection
        if current_robot_task == Task.GO_TO_COLLECTION:
            # the robot already reaches the collection bin
            if np.linalg.norm(x[i, :2] - current_goal_loc[i]) < distance_threshold:
                is_treasure_picked_up[i] = False
                is_task_available[current_treasure_list[i]] = True
                current_treasure_list[i] = None
                current_task_list[i] = Task.IDLE
                no_task_completed += 1
                current_goal_loc[i] = x[i, :2]

        # robot is going to recharge
        if current_robot_task == Task.GO_TO_RECHARGE:
            if np.linalg.norm(x[i, :2] - current_goal_loc[i]) < distance_threshold:
                is_recharging[i] = True
                current_task_list[i] = Task.RECHARGING

        # robot is recharging
        if current_robot_task == Task.RECHARGING:
            battery_level[i] = energy_gain(battery_level[i])
            if battery_level[i] >= 80:
                current_task_list[i] = Task.IDLE
                recharging_available[recharging_loc2ind[tuple(current_goal_loc[i])]] = True
                current_goal_loc[i] = x[i, :2]

        # robot is idle?
        if current_robot_task == Task.IDLE:
            if battery_level[i] > 30.0:
                current_task_list[i] = Task.GO_TO_PICK
                target_task_loc = task_assign(np.array([task_loc_list[j] for j in range(5) if is_task_available[j]]), np.array(x[i, :2])[None, :])[0]
                current_goal_loc[i] = target_task_loc
                is_task_available[task_loc2ind[tuple(target_task_loc)]] = False
                current_treasure_list[i] = task_loc2ind[tuple(target_task_loc)]
            else:
                available_recharger_index = np.where(recharging_available)[0]
                if np.size(available_recharger_index) == 0:
                    current_task_list[i] = Task.WAIT_FOR_RECHARGING
                    current_goal_loc[i] = x[i, :2]
                else:
                    current_task_list[i] = Task.GO_TO_RECHARGE
                    current_goal_loc[i] = recharging_loc[available_recharger_index[0]]
                    recharging_available[available_recharger_index[0]] = False

    prev_battery = battery_level
    battery_level = energy_loss(prev_battery, dist_travel, is_treasure_picked_up)
    prev_x = x

    dxu = unicycle_position_controller(x.T, current_goal_loc.T)
    # dxu = uni_barrier_cert(dxu, x.T)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    if np.sum(is_alive) == 0:
        print('All robots are dead!')
        break
