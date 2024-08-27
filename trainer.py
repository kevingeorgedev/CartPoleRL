import torch
import torch.nn as nn
import torch.optim as optim
from replay import Transition
    
class DuelDQNet(nn.Module):
    def __init__(self, n_observations, n_actions, use_attention) -> None:
        super(DuelDQNet, self).__init__()

        self.use_attention = use_attention

        self.n_actions = n_actions
        self.n_observations = n_observations

        self.linear_seq = nn.Sequential(
            nn.Linear(n_observations, 128, bias=False),
            nn.ReLU(),

            nn.Linear(128, 256, bias=False),
            nn.ReLU(),

            nn.Linear(256, 512, bias=False),
            nn.ReLU(),

            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
        )

        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=16, batch_first=True)

        self.fc_adv = nn.Linear(256, n_actions)
        self.fc_val = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear_seq(x)

        if self.use_attention:
            x = x.unsqueeze(1)
            attn_output, _ = self.attn(x, x, x)
            x = attn_output.squeeze(1)

        adv: torch.Tensor = self.fc_adv(x)
        adv = adv.view(-1, self.n_actions)
        val: torch.Tensor = self.fc_val(x)
        val = val.view(-1, 1)
        
        qval = val + (adv - adv.mean(dim=1, keepdim=True))
        return qval
    
    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action[0].item()

class Network:
    def __init__(self, lr, n_observations, n_actions, gamma, device, use_attention) -> None:
        self.device = device
        self.policy = DuelDQNet(n_observations, n_actions, use_attention).to(self.device)
        self.target = DuelDQNet(n_observations, n_actions, use_attention).to(self.device)
        self.criterion = nn.MSELoss()
        self.gamma  = gamma
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)

    def train(self, batch: Transition):
        states      = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions     = torch.Tensor(batch.action).float()
        rewards     = torch.Tensor(batch.reward)
        masks       = torch.Tensor(batch.mask)

        logits      = self.policy(states).squeeze(1)
        next_logits = self.target(next_states).squeeze(1)
        
        logits = torch.sum(logits.mul(actions), dim=1)
        target = rewards + masks * self.gamma * next_logits.max(1)[0]

        loss = self.criterion(logits, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss