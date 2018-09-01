import random
import numpy as np
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed = 777):
        self.buffer_size  = buffer_size
        self.random_seed = random_seed
        self.buffer = deque()
        self.count = 0
        random.seed(self.random_seed)

    def size(self):
        return self.count

    def add(self, history, action, reward, terminate):
        # In history , [:, :, :, :-1] is s1 , [ :, :, :, :, 1:] is s2
        # deque 에 넣기 위해서 list 형태로 input들을 합쳐준다.
        memory = (history[:, :, :, : -1], action, reward, terminate, history[:,:,:,1:])

        if self.count + 1 < self.buffer_size:
            self.buffer.append(memory)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(memory)


    def minibatch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s1 = np.array([ x[0] for x in batch ])
        action = np.array([ x[1] for x in batch])
        reward = np.array([ x[2] for x in batch])
        terminate = np.array([ x[3] for x in batch])
        s2 = np.array([ x[4] for x in batch])

        return s1, action, reward, terminate, s2

    def clear(self):
        self.buffer.clear()
        self.count = 0