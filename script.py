import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from enum import Enum
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches


np.random.seed = 1024


class Task(Enum):
    IDLE = 0
    GO_TO_PICK = 1
    GO_TO_COLLECTION = 2
    GO_TO_RECHARGE = 3
    RECHARGING = 4
    WAIT_FOR_RECHARGING = 5
    DEAD = 6

# TODO communication graph
# Instantiate Robotarium object
N = 5
# initialization_loc = np.random.uniform(low=-2, high=-2, size=(3, N))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

r.chassis_patches[0].set_facecolor("#FF0000")
r.chassis_patches[1].set_facecolor("#FFA500")
r.chassis_patches[2].set_facecolor("#FFFF00")
r.chassis_patches[3].set_facecolor("#008000")
r.chassis_patches[4].set_facecolor("#0000FF")

task1_point = patches.Circle((0, 0), radius=0.03, color="#000000")
task2_point = patches.Circle((0, 0.8), radius=0.03, color="#000000")
task3_point = patches.Circle((0, -0.8), radius=0.03, color="#000000")
task4_point = patches.Circle((0.8, 0), radius=0.03, color="#000000")
task5_point = patches.Circle((-0.8, 0), radius=0.03, color="#000000")
collection_point = patches.Circle((-1.3, -0.8), radius=0.03, color="#696969")
recharger1_point = patches.Circle((-1.3, 0.8), radius=0.03, color="#000000")
recharger2_point = patches.Circle((1.3, 0.8), radius=0.03, color="#000000")

task_points = [task1_point, task2_point, task3_point, task4_point, task5_point]
recharger_points = [recharger1_point, recharger2_point]

r.axes.add_patch(task1_point)
r.axes.add_patch(task2_point)
r.axes.add_patch(task3_point)
r.axes.add_patch(task4_point)
r.axes.add_patch(task5_point)
r.axes.add_patch(collection_point)
r.axes.add_patch(recharger1_point)
r.axes.add_patch(recharger2_point)

robot_color = {0: 'Red', 1: 'Orange', 2: 'Yellow', 3: 'Green', 4: 'Blue'}

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
current_treasure_list = [100] * 5

# How many iterations do we want (about N*0.033 seconds)
iterations = 10000
low_battery = 30
enough_battery = 60


def energy_loss(prev_energy, dist, is_treasure, alpha=0.03, beta=2, gamma=0.1):  # TODO alpha, beta
    result = prev_energy - (alpha + beta * dist + gamma * is_treasure)
    return np.where(result < 0, 0, result)


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
# unicycle_position_controller = create_si_position_controller()

prev_x = r.get_poses().T.copy()
goals = prev_x[:, :2]
prev_battery = battery_level

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()
bar = ax.bar(range(5), battery_level)
ax.hlines([low_battery, enough_battery], xmin=-.4, xmax=4.4, colors="000000")


def autolabel(bar_plot):
    texts = []
    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        texts.append(ax.text(rect.get_x() + rect.get_width()/2., height + 3,
                     battery_level[idx], ha='center', va='bottom', rotation=0))
    return texts


bar_text = autolabel(bar)

for index in range(5):
    bar[index].set_color(robot_color[index])
# bar[0].set_color()
bars = ('1', '2', '3', '4', '5')
r.step()

for k in (range(iterations)):
    x = r.get_poses().T
    dist_travel = np.linalg.norm(x[:, :2] - prev_x[:, :2], axis=1)
    for i in range(N):
        current_robot_task = current_task_list[i]

        # robot is dead?
        if current_robot_task == Task.DEAD:
            continue
        if battery_level[i] <= 0:
            print(f'Robot {i} is dead! Last task is {current_task_list[i]} at {current_goal_loc[i]}, current location is {x[i, :2]}')
            if current_robot_task == Task.GO_TO_RECHARGE:
                recharging_available[recharging_loc2ind[tuple(current_goal_loc[i])]] = True
                recharger_points[recharging_loc2ind[tuple(current_goal_loc[i])]].set_color("000000")
            elif current_robot_task == Task.GO_TO_PICK or current_robot_task == Task.GO_TO_COLLECTION:
                is_task_available[current_treasure_list[i]] = True
                task_points[current_treasure_list[i]].set_color("000000")
            current_task_list[i] = Task.DEAD
            current_goal_loc[i] = x[i, :2]
            is_alive[i] = False
            continue

        # robot is waiting for recharging?
        if current_robot_task == Task.WAIT_FOR_RECHARGING:
            available_index = np.where(recharging_available)[0]
            if np.size(available_index) == 0:
                continue
            current_goal_loc[i] = recharging_loc[available_index[0]].copy()
            recharging_available[available_index[0]] = False
            current_task_list[i] = Task.GO_TO_RECHARGE
            recharger_points[available_index[0]].set_color(robot_color[i])

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
                task_points[current_treasure_list[i]].set_color("000000")
                current_treasure_list[i] = 100
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
            if battery_level[i] >= enough_battery:
                current_task_list[i] = Task.IDLE
                recharging_available[recharging_loc2ind[tuple(current_goal_loc[i])]] = True
                recharger_points[recharging_loc2ind[tuple(current_goal_loc[i])]].set_color("000000")
                current_goal_loc[i] = x[i, :2]

        # robot is idle?
        if current_robot_task == Task.IDLE:
            if battery_level[i] > low_battery:
                current_task_list[i] = Task.GO_TO_PICK
                target_task_loc = task_assign(np.array([task_loc_list[j] for j in range(5) if is_task_available[j]]), np.array(x[i, :2])[None, :])[0]
                current_goal_loc[i] = target_task_loc.copy()
                is_task_available[task_loc2ind[tuple(target_task_loc)]] = False
                current_treasure_list[i] = task_loc2ind[tuple(target_task_loc)]
                task_points[task_loc2ind[tuple(target_task_loc)]].set_color(robot_color[i])
            else:
                available_recharger_index = np.where(recharging_available)[0]
                if np.size(available_recharger_index) == 0:
                    current_task_list[i] = Task.WAIT_FOR_RECHARGING
                    current_goal_loc[i] = x[i, :2]
                else:
                    current_task_list[i] = Task.GO_TO_RECHARGE
                    current_goal_loc[i] = recharging_loc[available_recharger_index[0]]
                    recharging_available[available_recharger_index[0]] = False
                    recharger_points[available_recharger_index[0]].set_color(robot_color[i])

    prev_battery = battery_level
    battery_level = energy_loss(prev_battery, dist_travel, is_treasure_picked_up)

    for index, level in enumerate(battery_level):
        bar[index].set_height(level)
        bar_text[index].set_y(level + 3)
        bar_text[index].set_text(str(round(level, 1)))
    # ax.text(range(5), battery_level + 3, [str(i) for i in battery_level])
    # fig.canvas.draw()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    # plt.pause(0.1)
    # fig.clear()

    prev_x = x.copy()

    dxu = unicycle_position_controller(x.T, current_goal_loc.T)
    dxu = uni_barrier_cert(dxu, x.T)  # TODO collision avoidance

    r.set_velocities(np.arange(N), dxu)
    r.step()
    # TODO check task status

    if np.sum(is_alive) == 0:
        print('All robots are dead!')
        break
