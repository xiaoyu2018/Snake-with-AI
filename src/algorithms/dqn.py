from torch import nn
from .model import Model
import torch

class LinerNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu=nn.ReLU()
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x=self.relu(x)
        x = self.fc2(x)
        return x


class DQN(Model):
    def __init__(self):
        super().__init__()
        self.net = LinerNet(9, 3)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()
        self.gama = 0.9
        
    def predict(self, state):
        state=torch.tensor(state,dtype=torch.float)
        out=self.net(state)
        print(out)
        return torch.argmax(out).item()

    def train_step(self, state, reward):
        self.optimizer.zero_grad()
        output = self.net(state)
        loss = self.loss_func(output, reward)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        torch.save(self.net.state_dict(), 'dqn.pkl')

    def load(self,param_path):
        self.net.load_state_dict(torch.load(param_path))


if __name__=='__main__':
    a=torch.tensor([1,2,3,4,5,6,7,8,9,10,11])
    print(torch.argmax(a).item())