from replay import ReplayMemory
from trainer import Network

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Agent():
    def __init__(self, memory_size, lr, n_observations, n_actions, gamma, device, use_attention):
        self.replay = ReplayMemory(capacity=memory_size)
        self.trainer = Network(lr=lr, n_observations=n_observations, n_actions=n_actions, gamma=gamma, device=device, use_attention=use_attention)

    def get_action(self, state):
        return self.trainer.policy.get_action(input=state)