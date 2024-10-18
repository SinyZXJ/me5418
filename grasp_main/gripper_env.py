import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from utils import calculate_distance, calculate_reward  # 从 utils 引入函数

class GripperEnv(gym.Env):
    def __init__(self):
        super(GripperEnv, self).__init__()

        # Start pybullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Utilizing pybullet_data
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.full(12, -10), high=np.full(12, 10), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # Seed the environment
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.81)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, -0.65])

        robot_ids = p.loadSDF("models/gripper/wsg50_one_motor_gripper_new.sdf")
        self.robot_id = robot_ids[0]

        # Load a random object
        self.object_id = p.loadURDF("random_urdfs/000/000.urdf", [0.5, 0.0, 0.0])
        self.object_initial_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        return self._get_observation(), {}


    def step(self, action):
        # Action: left and right both considered
        for joint in range(7, 10, 2):  # Refers to 7,9
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL, targetPosition=action[int((joint-7)/2)])

        p.stepSimulation()
        time.sleep(1 / 10.0)

		# metrics
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # global pos
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # l&r finger pos
        left_finger_pos = p.getLinkState(self.robot_id, 7)[0]  # 左爪子
        right_finger_pos = p.getLinkState(self.robot_id, 9)[0]  # 右爪子

        # target pos
        object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        observation = np.concatenate([gripper_pos, left_finger_pos, right_finger_pos, object_pos]).astype(np.float32)
        print("Observation:", observation)  # 打印观测值
        print("Observation Shape:", observation.shape)  # 打印观测值的形状
        return observation


    def _calculate_reward(self):
        end_effector_pos = p.getLinkState(self.robot_id, 7)[0]
        object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        return calculate_reward(end_effector_pos, object_pos, self.object_initial_pos)

    def _check_done(self):
        object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        if object_pos[2] > 0.2:
            return True
        return False

    def close(self):
        p.disconnect(self.physics_client)
