'''
    dqn--深度Q网络（Q-learning+神经网络），神经网络代替了Q-learning中的QTable，就是把QTable中离散的数据拟合成一个神经网络（多元函数）
    经典dqn还有两个trick：experience replay和固定Q-target
    experience replay：将每一次行动的环境信息和反馈信息记录到经验池（于agent.py实现）
    固定Q-target：DQN中会有两个结构完全相同但是参数却不同的网络，一个用于预测Q估计（MainNet），一个用于预测Q现实（target），MainNet使用最新的参数，target会使用很久之前的参数
'''

from torch import nn
import torch
from .base import Model

class LinerNet(nn.Module):
    '''
    输入神经元个数即状态，输出神经元即设置的action个数，数值含义为在当前状态下做出某个action将会获得的value。
    value可看作长期reward。reward仅与单步决策相关，而value考虑的更加长远。
    '''
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
        # 损失=E[(真实value−预测value)^2]
        self.loss_func = nn.MSELoss()
        self.gama = 0.85
        

    def predict(self, state):
        '''依据当前环境信息，获取神经网络决策'''
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

        # 预测value
        out = self.net(state)
        label=out.clone()
        
        for i in range(len(label)):
            # 近似真实value，作为标签
            label[i][action[i]]=reward[i]+self.gama*torch.max(self.net(next_state[i]),0).values.item()


        self.optimizer.zero_grad()
        loss = self.loss_func(out, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        '''存储LinerNet模型参数至当前工作目录下'''
        torch.save(self.net.state_dict(), 'dqn.pkl')

    def load(self,param_path):
        '''从指定param_path读取模型加载至LinerNet'''
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