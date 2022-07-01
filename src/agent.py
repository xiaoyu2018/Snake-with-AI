from game import Game
from algorithms.model import Model
import random

class Agent:

    def __init__(self,game:Game,model:Model) -> None:
        self.game=game
        # 估值函数
        self.model=model
        
        self.gama = 0.9
        self.epsilon = 0
        # 记录当前游戏轮数
        self.current_epoch=0


    def get_action(self):
        ...
    
    def get_state(self):
        ...
    
    def train_long_memory(self):
        ...
    
    def train_short_memory(self):
        ...

    def play(self):
        ...