# Addestramento per Rinforzo di un Agente in Ambienti Procedurali / Reinforcement Learning of an Agent in Procedural Environments
<img src="agent training.gif" height="300" width="300">
The following work was made for a joint Bachelor's Degree Thesis in Computer Science @ University of L'Aquila A.Y. 2017-2018. The first
part consists in a Random Procedural Environment Generator through Constraint Programming (CLPR library from Prolog); the second part
transforms the generated environments in learning environments using Pygame for the simulation and Keras for the agent.

Here are a few keypoints to help you navigate the prototype code:

### Observation
The observation simulates a set of 40 laser proximity sensors (geometrically, they are treated as segments starting from the center of the
robot which collide with the environment generating intersection points which are then given to the agent) and has the shape of an array
of tuples each containing 3 informations:
### (angle, distance, isObject)
- Angle: the angle of the direction the sensor was facing, relative to the environment ([0,359]);
- Distance: the distance between the agent and the eventual point of intersection found (a high number is set instead if nothing was hit);
- isObject: boolean flag telling the agent whether the eventual point hit an objective or not (assumes the agent can recognize objectives);

### Actions
There are 3 possible actions: rotate right by 45°, rotate left by 45° or move in the faced direction.

### Task & Reward
The task is to collect as many objectives as possible within 1500 time steps (approximately 15 seconds). The agent provided in the code
succeedes in completing the task and definitely outperforms a human player. Reward is 1 for every objective collected.

### Future Developments
The first expansion of this work would be to complete the above mentioned task moving the objectives spawn points inside the environment
rooms. This already seems to require a formulation for curiosity as seen in a recent paper published by DeepMind. 
(https://github.com/google-research/episodic-curiosity) Next, once the agent is comfortable exploring a single environment while finding
objectives, one could attempt to train it in a set of 20-30 environments (switching it with each new episode) until it's capable of
performing the same task on a new one which it has never seen before.

### Installation
- make sure you have clean python paths, taking care of the `.local` directory as well.
- make sure to delete all `__pycache__` and other temporary files from your project.
- make sure you do not have conflicting conda installations.
- clone the repo.
- start a terminal inside the cloned directory.
- ` curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh `
- ` bash Miniconda3-latest-Linux-x86_64.sh `
- ` conda create --name tf python=3.9 `
- ` conda deactivate `
- **Restart the terminal**
- ` conda activate tf  `
- ` conda install -c conda-forge cudatoolkit=11.8.0 `
- ` pip install nvidia-cudnn-cu11==8.6.0.163 `
- ` CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) `
- ` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib `
- ` mkdir -p $CONDA_PREFIX/etc/conda/activate.d `
- ` echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
- ` echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh `
- ` pip install --upgrade pip `
- ` pip install tensorflow==2.12.* `
- ` python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" `
- ` rm -rf Miniconda3-latest-Linux-x86_64.sh `
- ` pip install -r requirements.txt `
