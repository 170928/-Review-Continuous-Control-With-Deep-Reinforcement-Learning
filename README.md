# -Review-Continuous-Control-With-Deep-Reinforcement-Learning

> 의역과 오역이 있습니다. 오역에 대해서는 언제나 지적해주세요.

## Abstact
이 논문은 DQN에서 사용되는 기법을 기반으로 continuous action domain에 RL을 적용시키는 아이디어를 제시합니다.  
"actor-critic, model-free" 알고리즘을 제시하며, 이 알고리즘은 "deterministic policy graident"를 기반으로 작동합니다.  
이 알고리즘은 cart pole swing-up, dexterous manipulation, legged locomotion and car driving 과 같은 문제에 효과적으로 작동함을 실험을 통해서 입증하였습니다.  
그리고, raw pixel로부터 정책(policy)를 학습할 수 있는 "end to end" 가 가능하다는 것을 보여줍니다.

## Introduction

