from game import Game
# from algorithms.base import Model
# from algorithms.dqn import DQN
from algorithms import *
import random
from collections import deque
from utils import Direction,GRID_NUM,PARAM_PATH

class Agent:

    def __init__(self,game:Game,model:Model) -> None:
        self.game=game
        # 估值函数模型
        self.model=model
        # epsilon贪心策略的概率
        self.epsilon = 200
        # 记录当前游戏轮数
        self.current_epoch=1
        # 经验池(上限设为500)
        self.memory=deque(maxlen=800)
        # 参数保存间隔轮数
        self.n_save=25

    def get_action(self,train=False):
        '''
        0:沿当前方向前进
        1:左转
        2:右转
        '''
        if(not train):
            self.model.load(PARAM_PATH)
            self.epsilon=0
        elif(self.epsilon>50):
            self.epsilon-=self.current_epoch*0.01

        sample=random.randint(0,500)
        cur_dir=int(self.game.snake.direction)
        # 随机选择actoin
        if(sample<self.epsilon):
            action=random.randint(0,2)

        # 神经网络选择action
        else:
            action=self.model.predict(self.get_state())
        
        
        #解析action
        if(action==0):
            return action,Direction(cur_dir)
        elif(action==1):
            return action,Direction((cur_dir-1)%4)
        elif(action==2):
            return action,Direction((cur_dir+1)%4)

    def get_state(self):
        '''
        状态设置为三个方向(前/左/右)上是否有危险、食物与蛇头的相对位置、当前行进方向
        使用1x9的向量表示
        '''
        info=self.game.get_env_info()
        
        def _get_new_head(pos,direction:Direction):
            
            if(direction==Direction.UP):
                pos=(pos[0],pos[1]-1)
            elif(direction==Direction.DOWN):
                pos=(pos[0],pos[1]+1)
            elif(direction==Direction.LEFT):
                pos=(pos[0]-1,pos[1])
            elif(direction==Direction.RIGHT):
                pos=(pos[0]+1,pos[1])
            return pos
        
        is_up=info['snake_direction']==Direction.UP
        is_down=info['snake_direction']==Direction.DOWN
        is_left=info['snake_direction']==Direction.LEFT
        is_right=info['snake_direction']==Direction.RIGHT
        
        return [

            # 行进方向
            is_up,
            is_down,
            is_left,
            is_right,

            # 食物与蛇头相对位置(简单归一化)
            (info['food_pos'][0]-info['snake_body'][0][0])/GRID_NUM[0],
            (info['food_pos'][1]-info['snake_body'][0][1])/GRID_NUM[1],

            # 判断碰撞
            # forward
            (is_up and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.UP))) or
            (is_down and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.DOWN))) or
            (is_left and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.LEFT))) or
            (is_right and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.RIGHT))),
            #left
            (is_up and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.LEFT))) or
            (is_down and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.RIGHT))) or
            (is_left and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.DOWN))) or
            (is_right and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.UP))),
            #right
            (is_up and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.RIGHT))) or
            (is_down and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.LEFT))) or
            (is_left and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.UP))) or
            (is_right and self.game.snake.is_danger(_get_new_head(info['snake_body'][0],Direction.DOWN))),
        ]
    
    def train_long_memory(self):
        if len(self.memory) != 0:
            states, actions, rewards, next_states = zip(*(self.memory))
            self.model.train_step(states, actions, rewards, next_states)
    
    def train_short_memory(self, state, action, reward, next_state):
        if reward == 0:
            if random.randint(1,15) == 1: 
                self.memory.append([state, action, reward, next_state])
        else:
            self.memory.append([state, action, reward, next_state])
        self.model.train_step([state], [action], [reward], [next_state])

    def train(self):
        
        while(True):
            if not self.current_epoch % self.n_save:
                    self.model.save()
            

            state=self.get_state()
            action=self.get_action(True)
            print(action)
            reward,over,score=self.game.play_step(action[1])
            next_state=self.get_state()

            self.train_short_memory(state,action[0],reward,next_state)
            self.train_long_memory()
            print("======================================")
            print(f'epoch:{self.current_epoch},score:{score}\n')
            self.current_epoch+=1
            if(over):
                self.game.reset()
            import pygame
            # pygame.time.delay(1000)
    def auto_play(self):
        import pygame
        while(True):
            dir=self.get_action(train=False)[1]
            self.game.play_step(Direction(dir))
            # print(self.get_state())
            pygame.time.delay(250)

if __name__=='__main__':
    agent=Agent(Game(),DQN())
    # agent.train()
    agent.auto_play()
    # # info=agent.game.get_env_info()

    # # print(info['food_pos'][0]-info['snake_body'][0][0])
    # # print(GRID_NUM)
    # # print(agent.get_state())
    # agent.auto_play()
    
    
