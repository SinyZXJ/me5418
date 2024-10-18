# OpenAI Gym Report

This project simulates a robotic gripper using PyBullet and Gymnasium, allowing for reinforcement learning algorithms (such as PPO) to be applied in a simulated environment. The goal is to control a robotic gripper to reach and grasp an object, overcoming challenges such as avoiding obstacles and working within dynamic environments.

### Prerequisites

Before running the project, ensure you have fully installed requirements listed in environment.yaml.

### Installation

1. Download the repository from CANVAS.

2. Create the conda environment using environment.yaml

   ```
   conda env create -f environment.yaml
   ```

3. Activate the environment:

   ```
   conda activate gripper_env
   ```

### Visulization

To test the environment is actually built, run the simulation as below:

1. Navigate the the directory

   ```
   cd path_to_the_directory
   ```

2. run the test script

   ```
   python environment_test.py
   ```

   