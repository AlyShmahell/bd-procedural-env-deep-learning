import tensorflow as tf
import numpy as np
from collections import deque
import random

class SLAMAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.temp_memory = deque(maxlen=200)
        self.gamma = 1
        self.epsilon = 1.0
        self.epsilon_min = 0.025
        self.epsilon_decay = 0.996
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.01
        self.random_actions = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.model = self._build_model()
        self.reward = []

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=25, activation='relu', input_shape=(self.state_size, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=50, activation='relu'),
            tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=50, activation='relu'),
            tf.keras.layers.Dense(units=25, activation='relu'),
            tf.keras.layers.Dense(units=self.action_size, activation='softmax')])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        return model

    def remember(self, state, action, reward, next_state, done):
        if reward == 0:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.temp_memory.append((state, action, reward, next_state, done))

    def act(self, state):
        current_min_distance = 1
        for i in range(len(state[0])):
            if state[0][i][1] < current_min_distance and not state[0][i][2]:
                current_min_distance = state[0][i][1]
        if current_min_distance < 0.049:
            return random.randrange(self.action_size - 1), False
        if np.random.rand() <= self.epsilon:
            if state[0][18][2] or state[0][19][2] or state[0][20][2]:
                return 2, False
            return self.random_actions[random.randrange(len(self.random_actions) - 1)], True
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]), False

    def replay(self, batch_size):
        minibatch = []

        if len(self.temp_memory) > 0:
            minibatch.extend(self.temp_memory)
            self.memory.extend(self.temp_memory)
            self.temp_memory = []

        if len(self.memory) > batch_size:
            minibatch.extend(random.sample(self.memory, batch_size - len(minibatch)))
        else:
            minibatch.extend(random.sample(self.memory, len(self.memory) - len(minibatch)))

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=10, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print((f"REPLAY actual epsilon: {str(self.epsilon)}"))

    def save(self, fn):
        self.model.save(fn)
        print(f"actual epsilon: {str(self.epsilon)}")

    def load(self, name, last_random_value):
        self.model.load_weights(name)
        self.epsilon = last_random_value
