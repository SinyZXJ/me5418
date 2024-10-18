import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class GripperEnv(gym.Env):
    def __init__(self):
        super(GripperEnv, self).__init__()

        # Start pybullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Utilizing pybullet_data

        # Action Space:4 actions (4 DoF)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1]), dtype=np.float32)

        # Observatopm Space:End space + Object space
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -10, -10, -10]), 
            high=np.array([10, 10, 10, 10, 10]),
            dtype=np.float32
        )

        # Initilization
        self.reset()

    def reset(self):
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.81)

		# Load the objects needed
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0, 0, -0.65])

		# Load the robot
        robot_ids = p.loadSDF("models/gripper/wsg50_one_motor_gripper_new.sdf")
        self.robot_id = robot_ids[0]
        

		# Check joint info for debugging
        # num_joints = p.getNumJoints(self.robot_id)
        # for joint_index in range(num_joints):
        #     joint_info = p.getJointInfo(self.robot_id, joint_index)
        #     print(f"Joint {joint_index}: {joint_info}")

		# Load a random object
        self.object_id = p.loadURDF("random_urdfs/000/000.urdf", [0.5, 0.0, 0.0])

        return self._get_observation()


    def step(self, action):
        gripper_position_delta = action[:3]  # Gripper position change (x,y,z)
        current_position, current_orientation = p.getBasePositionAndOrientation(self.robot_id)
        new_position = np.array(current_position) + gripper_position_delta  # position uodate
        #p.resetBasePositionAndOrientation(self.robot_id, new_position, current_orientation)
       
        low_boundary = np.array([-0.5, -0.5, 0.3])  #workspace boundaries
        high_boundary = np.array([0.5, 0.5, 1.0])
        new_position = np.clip(new_position, low_boundary, high_boundary)

        # If the end effector has moved outside the boundaries, reset its position
        p.resetBasePositionAndOrientation(self.robot_id, new_position, current_orientation)

        p.stepSimulation()	# Key step
        time.sleep(1 / 2.0)

        # Action --> joint motion
        p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=0.2)  # left gripper
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.2)  # tight gripper

        p.stepSimulation()
        time.sleep(1 / 2.0)

        p.setJointMotorControl2(self.robot_id, 7, p.POSITION_CONTROL, targetPosition=-0.2)  # left gripper
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, targetPosition=-0.2)  # right gripper

        p.stepSimulation()
        time.sleep(1 / 2.0)
        
        # calculate reward and ending condition
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = False

        return observation, reward, done, {}

    def render(self, mode='human'):
        pass  # Open GUI

    def _get_observation(self):
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.robot_id)

        left_finger_pos = p.getLinkState(self.robot_id, 7)[0]
        right_finger_pos = p.getLinkState(self.robot_id, 9)[0]

        object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        return np.concatenate([gripper_pos, left_finger_pos, right_finger_pos, object_pos]).astype(np.float32)

    def _calculate_reward(self):
        # reward based on dist of endef and obj
        gripper_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        object_pos = p.getBasePositionAndOrientation(self.object_id)[0]

        distance = np.linalg.norm(np.array(gripper_pos) - np.array(object_pos))
        reward = -distance
        return reward

    def close(self):
        p.disconnect(self.physics_client)


if __name__ == '__main__':
    env = GripperEnv()
    obs = env.reset()

    for _ in range(2000):
        action = env.action_space.sample()  # Random action for it's just gym demonstration :)
        obs, reward, done, info = env.step(action)
        if done:
            break

    env.close()
