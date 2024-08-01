# Random Latent Exploration for Deep Reinforcement Learning

This is the codebase for "Random Latent Exploration for Deep Reinforcement Learning" (ICML '24). 

The ability to efficiently explore high-dimensional state spaces is essential for the practical success of deep Reinforcement Learning (RL). This paper introduces a new exploration technique called Random Latent Exploration (RLE), that combines the strengths of exploration bonuses and randomized value functions (two popular approaches for effective exploration in deep RL). RLE leverages the idea of perturbing rewards by adding structured random rewards to the original task rewards in certain (random) states of the environment, to encourage the agent to explore the environment during training. RLE is straightforward to implement and performs well in practice. To demonstrate the practical effectiveness of RLE, we evaluate it on the challenging Atari and IsaacGym benchmarks and show that RLE exhibits higher overall scores across all the tasks than other approaches, including action-noise and randomized value function exploration.


## Installation

### Clone this repository

```
git clone git@github.com:Improbable-AI/random-latent-exploration.git
```

### Create a conda environment

```
conda create -n rle python=3.8
```

### Installing IsaacGym

Follow the instructions in the [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) repository to setup IsaacGym.


### Installing IsaacGymEnvs

Once IsaacGym is installed and you are able to successfully run the examples, install IsaacGymEnvs:

```
cd isaacgym/IsaacGymEnvs
pip install -e .
```

### Install other requirements

```
pip install -r requirements.txt
```

### Running Experiments
To run RLE in Atari games, run the following:
```
python atari/ppo_rle.py
```

To run RLE in IsaacGymEnvs tasks, run the following:
```
python isaacgym/ppo_rle.py
```

## Acknowledgements

We thank [CleanRL](https://github.com/vwxyzjn/cleanrl) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) for their amazing work, which were instrumental in making this work possible.

## Citing

Please cite this work as:

```
@inproceedings{
    mahankali2024random,
    title = {Random Latent Exploration for Deep Reinforcement Learning},
    author = {Srinath V. Mahankali and Zhang-Wei Hong and Ayush Sekhari and Alexander Rakhlin and Pulkit Agrawal},
    booktitle = {Forty-first International Conference on Machine Learning},
    year = {2024}
}
```
