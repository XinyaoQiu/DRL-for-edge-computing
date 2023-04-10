# Deep Reinforcement Learning and Blockchain Mining for Mobile Edge Computing

This is a repository containing Python code and a corresponding article on the topic of mobile edge computing (MEC) and its optimization using deep reinforcement learning (DRL) techniques. The article discusses the challenges of designing an efficient task-offloading strategy for the whole MEC system, and proposes the use of multi-agent DRL to support smart task offloading in a MEC network.

Specifically, the project simplifies the MEC problem as video task processing and applies three different DRL methods based on Actor-Critic structure: Multi-Agent Advantage Actor-Critic (MAA2C), Multi-Agent Proximal Policy Optimization (MAPPO), and Multi-Agent Deep Deterministic Policy Gradient (MADDPG). The reward function for different environment parameters is compared, as well as the final results.

## Getting Started

To get started with this project, you can clone the repository and run the Python code on your machine. You will need to have Python 3 and the following packages installed:

- Tensorflow
- PyTorch
- Keras
- OpenAI Gym

You can install these packages using pip:

```python
pip install tensorflow keras gym torch
```

## Usage

The main code files in this repository are:

- `maa2c.py`: Implements the Multi-Agent Advantage Actor-Critic (MAA2C) algorithm.
- `mappo.py`: Implements the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.
- `maddpg.py`: Implements the Multi-Agent Deep Deterministic Policy Gradient (DDPG) algorithm.
- `env.py`: Defines the MEC environment and its reward function.
- `train.py`: Trains the agents using the specified DRL algorithm and environment parameters.
- `evaluate.py`: Evaluates the trained agents on the environment.

To train the agents, run `train.py` with the desired algorithm and environment parameters:

```
python train.py --algorithm maa2c --env-params env_params.json
```

To evaluate the trained agents, run `evaluate.py` with the same algorithm and environment parameters:

```
python evaluate.py --algorithm maa2c --env-params env_params.json
```

## References

1. X. Xiong, K. Zheng, L. Lei, and L. Hou, “Resource allocation based on deep reinforcement learning in iot edge computing,” IEEE J. Sel. Areas Commun., vol. 38, no. 6, pp. 1133–1146, 2020.
2. D. Nguyen, M. Ding, P. Pathirana, A. Seneviratne, J. Li, and V. Poor, “Cooperative task offloading and block mining in blockchain-based edge computing with multi-agent deep reinforcement learning,” IEEE Transactions on Mobile Computing, pp. 1–1, 2021.
3. A. Barto, R. Sutton, and C. Anderson, “Neuron like elements that can solve difficult learning control problems,” IEEE Transactions on Systems, Man, & Cybernetics, pp. 1–1, 1983.
4. Openai, “Openai baselines: Acktr a2c,” <https://openai.com/blog/baselines-acktr-a2c/>.
5. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017.
6. T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” 2016.
7. B. Yang, X. Cao, J. Bassey, X. Li, and L. Qian, “Computation offloading in multi-access edge computing: A multi-task learning approach,” IEEE Trans. Mob Comupt., pp. 1–1, 2021, doi:10.1109/TMC.2020.2990630.
8. Z. Shou, X. Lin, Y. Kalantidis, L. Sevilla-Lara, M. Rohrbach, S.-F. Chang, and Z. Yan, “DMC-Net: Generating discriminative motion cues for fast compressed video action recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 1268–1277.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
