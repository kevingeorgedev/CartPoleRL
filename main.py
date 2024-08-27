import gym
import time
import numpy as np
import random
from agent import Agent
import torch
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

print(f"Action space: {n_actions}")
print(f"Observation space: {n_observations}")

NUM_EPISODES = 500
GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 40000
BATCH = 128
LR = 2.5e-4
TRAIN_AFTER = 128
USE_ATTENTION = True
MOVING_AVERAGE = 10

UPDATE_STEPS = 100
train_mode = True

epsilon = INITIAL_EPSILON

agent = Agent(memory_size=REPLAY_MEMORY, lr=LR, n_observations=n_observations, n_actions=n_actions, gamma=GAMMA, device=device, use_attention=USE_ATTENTION)
learn_steps = 0
loss_fn = nn.MSELoss()
scores = []
mean_scores = []
total_score = 0

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    reward_e = 0

    for time_steps in range(200):
        env.render()
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            tensor_state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
            action = agent.trainer.policy.get_action(tensor_state)
        
        next_state, reward, done, _ = env.step(action)
        reward_e += reward

        agent.replay.add(state, action, next_state, reward, done)
        if len(agent.replay) > TRAIN_AFTER:
            if train_mode:
                print("starting training...")
                train_mode = False
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                agent.trainer.target.load_state_dict(agent.trainer.policy.state_dict())

            batch = agent.replay.sample(BATCH)
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*batch)

            batch_state = torch.tensor(batch_state, dtype=torch.float32, device=device)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32, device=device)
            batch_action = torch.tensor(batch_action, dtype=torch.float32, device=device).unsqueeze(1)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device).unsqueeze(1)
            batch_done = torch.tensor(batch_done, dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                policyQ_next = agent.trainer.policy(batch_next_state)
                targetQ_next = agent.trainer.target(batch_next_state)
                policy_max_action = torch.argmax(policyQ_next, dim=1, keepdim=True)
                y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, policy_max_action.long())

            loss = loss_fn(agent.trainer.policy(batch_state).gather(1, batch_action.long()), y)
            agent.trainer.optimizer.zero_grad()
            loss.backward()
            agent.trainer.optimizer.step()

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        
        if done:
            break
        state = next_state

    scores.append(reward_e)
    total_score += reward_e
    #mean_score = total_score / (episode + 1)
    #mean_scores.append(mean_score)
    sub_score = scores[MOVING_AVERAGE * -1:]
    mean_scores.append(np.mean(sub_score))

    print(f"Episode: {episode + 1}, Score: {reward_e}")

plt.plot(scores, label="Score")
plt.plot(mean_scores, label="Score MA(50)")
plt.xlabel("Episode #")
plt.ylabel("Score")
plt.legend(loc="upper left")
plt.show()