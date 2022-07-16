from torch import nn
import torch
from .base import Model

class LinerNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu=nn.ReLU()
        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN(Model):
    def __init__(self):
        super().__init__()
        self.net = LinerNet(9, 3)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()
        self.gama = 0.85
        

    def predict(self, state):
        state=torch.tensor(state,dtype=torch.float)
        out=self.net(state)
        # print(out)
        return torch.argmax(out).item()

    def train_step(self, state, action, reward, next_state):
        """
        state: 当前状态 2d
        action: 当前动作 1d
        reward: 当前奖励 1d
        next_state: 下一状态 2d
        """
        state=torch.tensor(state,dtype=torch.float)
        reward=torch.tensor(reward,dtype=torch.float)
        next_state=torch.tensor(next_state,dtype=torch.float)

        out = self.net(state)
        label=out.clone()
        
        for i in range(len(label)):
            label[i][action[i]]=reward[i]+self.gama*torch.max(self.net(next_state[i]),0).values.item()


        self.optimizer.zero_grad()
        loss = self.loss_func(out, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        torch.save(self.net.state_dict(), 'dqn.pkl')

    def load(self,param_path):
        self.net.load_state_dict(torch.load(param_path))


if __name__=='__main__':
    # a=torch.tensor([1,2,3,4,5,6,7,8,9])
    # print(torch.argmax(a).item())
    net=DQN()
    print(net.train_step(
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]],
        [0,1],
        [1,1],
        [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]])
        )