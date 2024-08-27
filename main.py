import gym
import time
import numpy as np
import random
from agent import Agent
import torch
import torch.nn as nn
from itertools import count

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

print(f"Action space: {n_actions}")
print(f"Observation space: {n_observations}")

NUM_EPISODES = 100
GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 40000
BATCH = 128
LR = 2.5e-4
TRAIN_AFTER = 128

UPDATE_STEPS = 100
train_mode = True

epsilon = INITIAL_EPSILON

agent = Agent(memory_size=REPLAY_MEMORY, lr=LR, n_observations=n_observations, n_actions=n_actions, gamma=GAMMA, device=device)
learn_steps = 0
loss_fn = nn.MSELoss()

for episode in count():
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

            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

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

    """if episode % UPDATE_STEPS == 0:
        # TODO: TRAIN AGENT
        pass"""

    print(f"Episode: {episode + 1}, Score: {reward_e}")


"""env.reset()
env.render()
time.sleep(5)
env.close()"""