# utils.py
import numpy as np

def calculate_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def calculate_reward(end_effector_pos, object_pos, object_initial_pos):
    distance = calculate_distance(end_effector_pos, object_pos)
    object_movement = calculate_distance(object_pos, object_initial_pos)

    if object_movement > 0.05:
        return 100 - distance
    return -distance
