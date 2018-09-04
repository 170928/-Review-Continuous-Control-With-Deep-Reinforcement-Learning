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
Stochastic environment 이기 때문에 state도 distribution으로 표현됩니다.  
뿐만 아니라, state transition dynamics 도 확률로써 표기됩니다.  
> Stochastic environment 에 대한 간단한 이해는 다음 주소를 참조하였습니다.  
> http://www.modulabs.co.kr/RL4RWS/18834
> Deterministic하다는 것은 어떤 함수가 무조건 parameter에 따라 결과가 결정된다는 의미이다.
즉, 같은 인자에 같은 결과가 나온다는 것. 앞서서 우리가 가정했던 Frozen Lake의 환경은
입력한 action 그대로 다음 state가 결정되는 상황이었다. 
그에 반해 non-deterministic한 환경에서는 입력과 관계없이 랜덤한 결과가 나온다.
gym에서 is_slippery 값을 True로 설정하게 되면 호수에서 움직일 때 미끄러져서 action과 무관하게
다음 state가 결정된다. 그렇기 때문에 이전 시간까지 우리가 Q-테이블을 채우는 방식을 그대로 가져오면
성공률이 현저하게 떨어지게 된다. Q-테이블은 내가 그대로 움직일 것을 가정하고 최대 reward 값을 리턴하는데
실제로 나는 그 방향대로 움직이지 않을 것이기 때문이다.
그렇다면 이런 상황에서 Q-테이블은 쓸모가 없어지는 것인가?
가끔씩 action대로 움직이는 경우도 발생하기 때문에 Q-table을 완전히 무시하는 것도 비효율적일 수 있다. 
그래서 제안하는 것이 Q-테이블을 보되, 업데이트할 때 다음 state의 reward를 적게 반영하는 것이다.
Q[state, action] = r + max(Q[new_state])
이 식은 내가 원하는대로 움직이는 것을 가정한 식이기 때문에 non-deterministic 상황에서는 Q[state] 값을 왜곡하므로
이 부분이 반영되는 비율을 적게 하고, 현재의 값을 많이 반영하면, Q-table 값을 어느정도 보정할 수 있다.  
Reward는 dicounted future reward 의 합계로써 정의됩니다.  

> ![image](https://user-images.githubusercontent.com/40893452/44576846-92a02000-a7ca-11e8-83c7-8e1f04c9ae46.png)   

Return (== Reward)는 선택되는 action에 따라 변하므로, action이 선택되고 행해지는 것이 stochastic 환경이기 때문에, Reward 또한
stochastic 하다.  

강화학습의 목적은 "state distribution" 에서 최대의 기대 수익 (expected return) 을 얻을 수 있는 정책 (policy)를 찾는 것입니다.  

> ![image](https://user-images.githubusercontent.com/40893452/44576977-e1e65080-a7ca-11e8-87d3-bc5004e8fd44.png)   

이 논문에서는 discounted state visitation distribution 을 다음과 같이 표기합니다.  

> ![image](https://user-images.githubusercontent.com/40893452/44577033-fde9f200-a7ca-11e8-8697-cb4a9b94718a.png)   

강화학습에서 사용하는 "action-value function"은 action a(t) 를 state s(t)에서 수행할 때 예상되는 수익을 의미하며 다음과 같이 표기할 수 있습니다.  

> ![image](https://user-images.githubusercontent.com/40893452/44577107-325dae00-a7cb-11e8-852d-9e39253ec7a0.png)  

강화학습을 사용할 때, "Bellman equation"으로 알려진 reculsive relationship을 이용합니다.  

> ![image](https://user-images.githubusercontent.com/40893452/44577336-cf204b80-a7cb-11e8-8e06-0686730311f7.png)  

만약 target policy 가 deterministic 하다면, bellman equation 내의 expectation을 없앨 수 있습니다.  

> ![image](https://user-images.githubusercontent.com/40893452/44577397-f7a84580-a7cb-11e8-9b54-9003dd35ed9d.png)  

이 식에 따르면, action-value function 이 target policy와 무관하게, 환경에 대해서만 이존하게 되므로 "off-policy" 접근으로 학습 할 수 있게 됩니다.  
> policy가 determinstic하기 때문에, policy에따라 변화될 수 있는 Expectation이 빠진 것을 알 수 있습니다. Deterministic policy를 가정하기 전의 수식에서는 a(t+1)을 선택하는 순간의 policy로 Q에 대한 Expection을 원래 구해야하기 때문에 off-policy가 아니지만, Determinsitic policy를 가정한다면 update 할 당시의 policy로 at+1를 구할 수 있기 때문에 off-policy가 됩니다. 
> https://reinforcement-learning-kr.github.io/2018/06/26/3_ddpg/ 참고  

네트워크의 parameter를 다음과 같이 표기합니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44577728-cc722600-a7cc-11e8-870d-e6689a9f1fe1.png)  

뉴럴 네트워크의 학습 과정에서 목적합수로 사용 될 loss function은 다음과 같이 정의됩니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44577792-eb70b800-a7cc-11e8-91c0-c7e9837e40e7.png)  

이때, y(t)는 다음과 같습니다.  
> ![image](https://user-images.githubusercontent.com/40893452/44577832-04796900-a7cd-11e8-9316-42f6d78ba5b3.png)

## Algorithm
Q-learning을 continous space에 적용하는 것은 불가능 합니다. continuous space에서 greedy policy를 찾는 방법은 매 time step 마다 a(t) 의 optimization이 수행되어야 함을 의미합니다.  
이 과정은 너무 느리기 때문에 실제로 적용하기에 무리가 있습니다.  
그러므로, DPG 알고리즘 비나의 actor-critic approach가 이 논문에서 제시됩니다.  

DPG 알고리즘은 parameterized actor function ![image](https://user-images.githubusercontent.com/40893452/44640718-43dacc00-a9fe-11e8-987e-18139f428540.png) 를 가지고 있습니다.  
![image](https://user-images.githubusercontent.com/40893452/44640718-43dacc00-a9fe-11e8-987e-18139f428540.png) 는 "current policy"를 의미하며, state를 action으로 매핑합니다.  
critic function ![image](https://user-images.githubusercontent.com/40893452/44640751-74226a80-a9fe-11e8-8f99-e9aeab5ca35f.png) 은 Q-learning에서 처럼 "Bellman equation"을 통해서 학습됩니다.  
actor 는 "start distribution ![image](https://user-images.githubusercontent.com/40893452/44640778-a9c75380-a9fe-11e8-973e-3f0275246560.png) 로 부터 기대되는 return에 chain rule을 적용해서 업데이트 됩니다. 즉, 아래와 같은 식으로 표현할 수 잇습니다.  
![image](https://user-images.githubusercontent.com/40893452/44640807-d3807a80-a9fe-11e8-88d8-0fb80534fcb8.png)  
> Silver et al.(2014) 에서 이 방법이 "policy gradient"라는 것을 증명했습니다.  

## Detail Algorithm
강화학습을 위해서 뉴럴 네트워크를 사용할 때 대부분의 최적화 알고리즘은 샘플들이 독립적이고 identical distribution을 따른다고 가정합니다.  
그러나, 실제로 샘프들이 연속된 탐색과정에서 발생하게 된다면 이 2가지 가정은 유지될 수 없습니다.  
즉, 샘플들 간에 "Correlation"이 존재하게 됩니다. 
추가적으로, 학습과정에서 "online" 방법 보다는 minibatch 를 활용한 "offline" 방법이 효과적입니다.  

"Replay buffer"은 exploration policy를 따라서 저장된 유한한 사이즈의 transition 저장소 입니다.  
replay buffer가 가득차게 되면 오래된 샘플들부터 버리고 새로운 데이터를 축적합니다.  
매 time step 마다, actor와 critic은 buffer로 부터 uniformly random하게 샘플링되는 미니배치를 통해서 업데이트 됩니다.  
"Soft" target update 방법을 통해서 직접적으로 weight 전체를 copy하는 방법이 아닌 조금씩 업데이트 하는 형태의 방식을 채택 하였습니다.  

(1) actor와 critic network의 복사본을 만듭니다. 
![image](https://user-images.githubusercontent.com/40893452/44642507-79d07e00-aa07-11e8-83a8-fe23b778cf7b.png)   
![image](https://user-images.githubusercontent.com/40893452/44642524-8b198a80-aa07-11e8-93dc-f4d8f1dc1e99.png)  
이들은 target value를 계산하기 위해서 사용됩니다.  
이 네트워크들의 가중치들은 leanred network를 천천히 따라가면서 업데이트 되도록 하는 "soft copy" 형태의 업데이트가 이루어집니다.  
이 방법은 학습과정의 안정성을 향상시킵니다.  

(2) ![image](https://user-images.githubusercontent.com/40893452/44642638-12ff9480-aa08-11e8-8c76-eb5ee89f6cae.png) 는 학습과정에서 발산의 문제 없이 "Critic"을 학습하기 위해서 안정적인 y(i) 를 가지기 위해 필요합니다. 즉, 2개의 네트워크가 모두 필요합니다.  

(3) low dimensional feature vector로부터 학습을 할 때, observation 대상들은 다른 environment에서 다른 범위 의 값을 가질 것입니다. 이는 다른 범위의 값을 가지는 환경들을 건너서 적절한 hyper-parameters를 찾는 것을 어렵게 합니다.  
이 문제를 해결하기 위해서 이 논문에서는 "batch-normalization"을 사용합니다. 그 결과, 환경과 요소에 관계없이 유사한 범위에 놓인 값을 통해서 학습이 이루어지게 됩니다.  
이 방법은 미니배치에서 샘플들 모두에서 각 dimension에 대해서 normalize를 수행합니다.  
그리고, 가깍의 레이어가 받는 입력값의 "covariance shift"를 최소화 시켜줍니다.  
이 논문에서는 ![image](https://user-images.githubusercontent.com/40893452/44643362-3841d200-aa0b-11e8-9113-b77e24bfacfd.png)
에 batch_normalization이 적용됩니다.  

(4) continuous action space 에서의 exploration 방법은 아래오 가타이 actor policy에 노이즈를 추가하는 방법으로 이루어지며, 이때, 노이즈는 환경에 따라 적절하게 변경됩니다.  
![image](https://user-images.githubusercontent.com/40893452/44643447-8ce54d00-aa0b-11e8-9392-29f3be00379c.png)  

![image](https://user-images.githubusercontent.com/40893452/44644154-09792b00-aa0e-11e8-9549-84ab0b0211f1.png)


## Result


![image](https://user-images.githubusercontent.com/40893452/44644167-19910a80-aa0e-11e8-9251-f9a2599183d1.png)

모든 작업에서 우리는 낮은 차원의 상태 설명 (예 : 관절 각 및 위치)과 환경의 고차원 렌더링을 모두 사용하여 실험을 실행했습니다.

 DQN (Mnih et al., 2013; 2015)에서와 같이 고 차원 환경에서 문제를 거의 완벽하게 관찰 할 수 있도록 조치 반복을 사용했습니다.

에이전트의 각 타임 스텝마다 시뮬레이션의 3 단계 타임 스텝을 수행하고 매번 에이전트의 동작과 렌더링을 반복합니다.

따라서 에이전트에게보고 된 관찰은 에이전트가 프레임 간의 차이를 사용하여 속도를 추론 할 수있게하는 9 개의 기능 맵 (3 가지 렌더링 각각의 RGB)을 포함합니다.

 프레임은 64x64 픽셀로 다운 샘플링되었고 8 비트 RGB 값은 [0,1]로 스케일 된 부동 소수점으로 변환되었습니다. 네트워크 구조 및 하이퍼 파라미터에 대한 자세한 내용은 보충 정보를 참조하십시오


 

> 공부하던 도중 facebook에서 서기호님이 작성해주신 요약 내용입니다. 간단하게 이해할 수 있어서 첨부하였습니다.
> DDPG =   
- Actor-Critic,   
- Model-Free,    
- Off-Policy   
- Deterministic   
- Policy Gradient  
이 논문에서 5개 단어만 알면 된다.  
x = observation   
s = state  
a = action  
Fully observable ->  
x=s  
pi, 
mu = policy  
deterministic policy: a = mu(s)  
결정이 되어 있다. s를 넣으면 a가 나온다  
stochastic policy: pi(a given s)  
확률적으로 나타내면 stochastic  
*deterministic는 끝났음  

markov decision process  

- initial state distribution p(s_1)  
- Transition Dynamics p(s_t+1 given s_t, a_t)  
모델을 다 알고 있으면 mdp를 만들수 있다  

- r(s,a) = reward function  
*model-free는 끝났음  

discounted future reward (return)  
return는 state, action의 함수이다. 즉, dynamics와 policy에 영향을 받는다.   
따라서 (stochastic policy의 경우) return는 stochastic 하다.  
R_t = r(s_0, a_0) + gamma * r(s_1, a_1) + gamma^2 * r(s_2, a_2)  
s_i는 e라는 distribution로 따른다. a_i는 pi라는 distribution 따른다.  
R은 Expectation으로 표시한다.  
J = Expectation [R_1]  
평균 취하는것 -> 배치 돌리는 느낌  
이 return을 maximize 하느 policy를 찾는것이 강화학습의 목적!!   
s와 a는 random variable  
Action-Value Function  
s_t에서 pi라는 policy를 따라해서 얻은 a_t라는 action를 취했을때의 expected return  
Q_pi (s_t, a_t) = E[R_t ㅣ s_t, a_t]  
Bellman Equation -> return은 recursive하게 구할수 있다  
Q_pi (s_t, a_t) = E[r(s_t,a_t) + gamma * E[Q_pi (s_t+1, a_t+1)]  
Q^mu (s_t, a_t) = Expectation[ r(s_t, a_t) + gamma * Q^u ( s_t+1, mu(s_t+1)]  
Q-learning 
mu(s) = argmax_a Q(s,a)   
deterministic policy  
Q^mu (s_t, a_t) = Expectation[ r(s_t, a_t) + gamma * Q^u ( s_t+1, mu(s_t+1)] <— sarsa 그리고 on-policy  
Q-function을 function approximator로  
s -> network -> Q-value 1, Q-value 2, Q-value 3  
DQN은 (s,a)가 discrete한 경우에만 가능  
이걸 continuous한 환경에서도 강화학습 할수 있게 만들어 보자  
Deep Deterministic Policy Gradient  
policy gradient: a= mu(s)를 하나의 네트워크 (theta_mu)로 보고, gradient descent로 optimal policy를 구하자  
gradient_theta^mu = E [ gradient_theta^mu Q(s,a given theta^ Q) ㅣ s=s_t, a=mu(s_t | mu) ]  
Q-function을 maximize하는 theta_mu를 찾아보자  
Actor-Critic  
gradient_theta^mu = E [ gradient_theta^mu Q(s,a given theta^ Q) ㅣ s=s_t, a=mu(s_t | mu) ]  
Q(s,a given theta^ Q) ㅣ s=s_t, a=mu(s_t | mu) = L  
E [ gradient_theta^mu Q(s,a given theta^ Q) ㅣ s=s_t, a=mu(s_t ) gradient_theta^mu(s | theta^mu) | s=s_t]  
theta_mu ~ dL/d theta_mu = dQ / da x da / d theta_mu  
dQ / da = critic  
da / d theta_mu = actor   
actor와 critic을 deep learning을 했다는게 이 논문의 포인트!!!!!   
experience reply를 썼고  
target network를 썼다. 딥러닝의 타겟 값을 다시 써서 했다.  
>

![ddpg_training](https://user-images.githubusercontent.com/40893452/45012288-68175800-b051-11e8-8ee0-ebe2cb133828.gif)








