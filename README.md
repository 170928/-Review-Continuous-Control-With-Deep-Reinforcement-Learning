# -Review-Continuous-Control-With-Deep-Reinforcement-Learning

> 의역과 오역이 있습니다. 오역에 대해서는 언제나 지적해주세요.

## Abstact
이 논문은 DQN에서 사용되는 기법을 기반으로 continuous action domain에 RL을 적용시키는 아이디어를 제시합니다.  
"actor-critic, model-free" 알고리즘을 제시하며, 이 알고리즘은 "deterministic policy graident"를 기반으로 작동합니다.  
이 알고리즘은 cart pole swing-up, dexterous manipulation, legged locomotion and car driving 과 같은 문제에 효과적으로 작동함을 실험을 통해서 입증하였습니다.  
그리고, raw pixel로부터 정책(policy)를 학습할 수 있는 "end to end" 가 가능하다는 것을 보여줍니다.

## Introduction
인공지능 분야의 중요한 목표 중 하나는 처리되지 않는 (unprocessed), 고 차원의 입력 데이터로부터 복잡한 작업을 해결해내는 것 입니다.  
최근 "Deep Q Netwrok" (DQN) 알고리즘은 처리되지 않은 raw pixel의 입력으로부터 사람보다 뛰어난 Atari Game 을 플레이 하는 것을 보여주었습니다.  
이때, 인공 신경망의 function approximation 능력은 강화학습의 "action-value function"을 대신하기위해서 사용됩니다.  
그러나, DQN은 고차원의 observation space 문제를 해결하였지만, action space 측면에서는 저 차원, discrete handle의 경우만을 처리할 수 있었습니다.  
문제는 대부분의 작업들이 continuous (real valued), high-dimensional 한 action space 를 가진다는 것입니다.  
DQN은 action-value fuction 을 최대화 하는 action을 찾는 것에만 집중하는 모델이기 때문에, 이러한 continous domain action에 적용되기 어렵습니다.  
Continuous action domains을 DQN에 적용하기 위해서 간단한 방법은 action space를 discretize 하는 것입니다.  
그러나. 이 방법은 "Curse of dimensionality"라는 한계를 갖게 합니다.  
large action space 는 효과적은로 학습과정에서 탐색하기 어려울 뿐더러, 성공적인 학습이 어렵게 됩니다.  
뿐만 아니라, action space의 discretization은 문제 해결에 중요한 정보인 action domain에 대한 구조적 정보를 사라지게 할 수 있습니다.  

이 논문에서 제안하는 알고리즘은 "model-free, off-policy actor-critic" 알고리즘입니다.  
neural network의 function approximation 특징을 활용해서 고차원의 continuous action space에서의 정책 (Policy) 를 학습합니다.  

제안하는 알고리즘은 deterministic policy gradient (DPG) 알고리즘을 기반으로 합니다. 그러나, 기존의 actor-critic method with neurla function approximators는 학습과정에서 안정적이지 못하다는 문제점이 있었습니다.  
그래서, 이 논문은 이전 DQN에서 제시한 방법을 적용시켜서 성능을 향상 시켰습니다.  
(1) Replay buffer를 통해서 학습 데이터가 갖는 correlation을 줄입니다.  
(2) off-policy 를 통해서 네트워크를 학습합니다.  
(3) target Q network를 사용하여 Temporal Difference backups동안 target 값을 일정하게 유지합니다.  
(4) Batch normalization 기법을 활용합니다.  

## Background
일반적인 Reinforement learning 환경을 고려합니다.  
environment ![image](https://user-images.githubusercontent.com/40893452/44575915-36d49780-a7c8-11e8-8c2d-505020fb3f8d.png) 와 상호작용하며, 각각의 timestep t 마다 agent는 observation ![image](https://user-images.githubusercontent.com/40893452/44575962-5a97dd80-a7c8-11e8-8f96-f11f79edefb9.png) 를 얻습니다.  
그에따라, action ![image](https://user-images.githubusercontent.com/40893452/44575996-70a59e00-a7c8-11e8-92fb-c47d402c6371.png) 과 reward를 받습니다.  
이 논문에서 고려하고자 하는 환경에서, action은 ![image](https://user-images.githubusercontent.com/40893452/44576050-903cc680-a7c8-11e8-9870-56d0beb84918.png) 가 됩니다.  
일반적으로 환경은 partially observed 되기 때문에 observation과 action 쌍의 모든 history를 모아서 하나의 state로 보아야 하지만, 이 논문에서는 fully observed라고 가정합니다. 그 의미는 아래와 같습니다.  
![image](https://user-images.githubusercontent.com/40893452/44576204-e7db3200-a7c8-11e8-9275-da9e6cb28f73.png)  
agent의 행동(behavior)는 정책 (policy) 에 의해서 결정됩니다.  
정책은 state를 action들의 확률 분포 (probability distribution) 으로 매핑합니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44576281-1d801b00-a7c9-11e8-9d28-c4fc9a34d784.png)  
환경 E 도 stochastic 합니다.  
그러므로, 이 모든 환경을 "Markov decision process"로 표현합니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44576430-7c459480-a7c9-11e8-8dc1-02a66698d89a.png)  








