# Model Based Reinforcement Learning Approach to HVAC optimal control

## Repo structure
### agent
- Multi-Objective MDP formulation with objectives as thermal comfort and energy consumption
- Lagrangian dual reinforcement learning approach
- fine tuning left to do
### mask-agent
- Single objective MDP of energy consumption, and thermal comfort enforced through hard constraint
- action bound approach
- in progress: Inferring change of environment to adjust the mask accordingly
### base
- single objective reinforcement learning formulation (electric cost), with demand response 
- demand response, Toronto Hydro electricity ToU (time of use) 
- To reduce HVAC actuation load, CAPS action smoothing utilized

## Algos implemented
- Rainbow DQN
- DQN
- SAC
- PPO
