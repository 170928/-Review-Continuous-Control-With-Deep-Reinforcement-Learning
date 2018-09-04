# [Reference Code]
# https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
# https://github.com/slowbull/DDPG
# https://github.com/liampetti/DDPG
# Thank you for your codes.


import tensorflow as tf
import os
import numpy as np
import gym
from Actor import Actor as actorNet
from Critic import Critic as criticNet

from replay_memory import ReplayBuffer
from Noise import Noise
from reward import Reward

# Maximum episodes run
MAX_EPISODES = 100000
# Max episode length
MAX_EP_STEPS = 50000
# Episodes with noise
NOISE_MAX_EP = 200
# Size of replay buffer
BUFFER_SIZE = 100000
# Size of batch
MINIBATCH_SIZE = 100


RANDOM_SEED = 777


# ====================================================
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5 # The rate of change (time)
SIGMA = 0.3 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate
# ====================================================

REWARD_FACTOR = 0.1
LEARNING_RATE = 0.00025
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ====================================================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
#ENV_NAME = 'CartPole-v0' # Discrete: Reward factor = 0.1
#ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.1
ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.01
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, noise, reward, discrete, saver, checkpoint_path):
    # Set up summary writer
    summary_writer = tf.summary.FileWriter("ddpg_summary")

    actor.update()
    critic.update()

    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Initialize noise
    ou_level = 0.

    for i in range(MAX_EPISODES):

        if i % 100 == 0:
            saver.save(sess, checkpoint_path)

        # s 는 environment에서 제공하는 첫번째 state 정보.
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        # buffer 초기화
        episode_buffer = np.empty((0,5), float)

        for j in range(MAX_EP_STEPS):

            # print(critic.w1.eval()[0,0])

            env.render()


            # a 는 actor의 current policy를 기반으로 예측한 q_value tensor [None x action_dim]
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))

            # stochastic environment에서의 e-greedy 를 위해서
            # Noise 를 추가한다.
            # 아랫 부분은 ornstein_uhlenbeck_level이라는 내용을 참조해야 합니다.. constant action space에서
            # 학습을 위해서 사용하는 방법이라고 합니다.
            if i < NOISE_MAX_EP:
                ou_level = noise.ornstein_uhlenbeck_level(ou_level)
                a = a + ou_level

                    # Set action for discrete and continuous action spaces
            if discrete:
                action = np.argmax(a)
            else:
                action = a[0]


            # 선택된 action을 기반으로 step을 진행시킨 후 결과를
            # 돌려받습니다.
            s2, r, terminal, info = env.step(action)

            # episode 내의 총 reward 값을 더합니다.
            ep_reward += r

            # ==========================================================================[중요한 부분]==============
            # Replay Buffer에 해당 정보를 더합니다.
            # episode_buffer라는 nparray에 [s, a, r, terminal, s2]의 배열을 넣어줍니다.
            episode_buffer = np.append(episode_buffer, [[s, a, r, terminal, s2]], axis=0)
            # ===================================================================================================

            # Replay Buffer에 Minibatch size 이상의 데이터가 담겨 있다면
            # 데이터를 가져와서 학습을 진행합니다.
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # critic을 통해서 Q(s, a)인 action-value function을 가져옵니다.
                # 이때 critic은 current policy를 평가하고 action-value function을 학습하므로,
                # actor의 예측 값을 가져옵니다.
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Critic이 가진 action_value function을 학습합니다.
                # 이때 필요한 데이터는 state batch, action batch, reward batch 입니다.
                # reward는 DQN (deepmind etc...) 논문에서 사용했던 것 과 같이
                # terminal 이라면 마지막 reward 자체.
                # terminal이 아니라면, s2 에서의 q_value 값에 discount factor를 곱한 값과 s 에서의 reward를 더한 값을
                # reward로 계사합니다.
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                # 모델의 성능 측정을 위해서 average Q 값을 확인합니다.
                # DQN (deepmind etc..) 논문에서 강화학습 모델의 진행도를 측정하기 위한 좋은
                # 지표로서 언급하였습니다.
                ep_ave_max_q += np.amax(predicted_q_value)

                # replay buffer에서 가져온 state_batch를 사용해서 actor의 current policy에 해당하는
                # q_value를 가져옵니다.
                # current_policy에 따른 q_value를 state 정보와 함께 넣어
                # Q(s, a)를 계산합니다 by critic
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)

                # grads 는 (1, BATCH_SIZE, ACTION_DIM) 의 배열이므로
                # (BATCH_SIZE, ACTION_DIM)의 입력을 받는 actor.train 함수를 위해서 grads[0]을 취합니다.
                # actor의 policy를 업데이트 하기 위해서 critic의 gradients를 받아와서 train합니다.
                actor.train(s_batch, grads[0])

                # actor와 critic network 모두 업데이트 합니다.
                actor.update()
                critic.update()

            # s2 를 s 로 바꾸어 진행합니다.
            s = s2

            if terminal:

                episode_buffer = reward.discount(episode_buffer)

                # Add episode to replay buffer
                for step in episode_buffer:
                    replay_buffer.add(np.reshape(step[0], (actor.state_dim,)), np.reshape(step[1], (actor.action_dim,)), step[2],
                                  step[3], np.reshape(step[4], (actor.state_dim,)))

                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(ep_reward))
                summary.value.add(tag='Perf/Qmax', simple_value=float(ep_ave_max_q / float(j)))
                summary_writer.add_summary(summary, i)

                summary_writer.flush()

                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_ave_max_q / float(j)))

                break


def main(_):

    with tf.Session() as sess:

        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        print("[Main start]")
        print("Env observation space ::", env.observation_space, env.observation_space.shape)
        print("Env action space ::", env.action_space, env.action_space.shape)

        state_dim = env.observation_space.shape[0]

        try:
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert (env.action_space.high == -env.action_space.low)
            discrete = False
            print('Continuous Action Space')
        except AttributeError:
            action_dim = env.action_space.n
            action_bound = None
            discrete = True
            print('Discrete Action Space')


        actor = actorNet(sess, state_dim, action_dim, action_bound, TAU)
        critic = criticNet(sess, state_dim, action_dim, TAU, actor.get_num_vars())

        SAVER_DIR = "./save/pend"
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(SAVER_DIR, "model")
        ckpt = tf.train.get_checkpoint_state(SAVER_DIR)



        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        reward = Reward(REWARD_FACTOR, GAMMA)


        try:
            if ckpt and ckpt.model_checkpoint_path:
                print("[Restore Model]")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("[Initialzie Model]")
                sess.run(tf.global_variables_initializer())

            train(sess, env, actor, critic, noise, reward, discrete, saver, checkpoint_path)
        except KeyboardInterrupt:
            pass

        if GYM_MONITOR_EN:
            env.monitor.close()


if __name__ == '__main__':
    tf.app.run()