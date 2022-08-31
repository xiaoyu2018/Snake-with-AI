from game import Game
# from algorithms.base import Model
# from algorithms.dqn import DQN
from algorithms import *
import random
from collections import deque
from utils import Direction,GRID_NUM,PARAM_PATH

class Agent:

    def __init__(self,game:Game,model:Model,train=False) -> None:
        self.game=game
        # 估值函数模型
        self.model=model
        # 控制epsilon贪心策略的概率
        self.epsilon = 200
        # 记录当前游戏轮数
        self.current_epoch=1
        # 经验池(上限设为500)
        self.memory=deque(maxlen=800)
        # 参数保存间隔轮数
        self.n_save=25

        if(not train):
            self.model.load(PARAM_PATH)
            self.epsilon=0


    def get_action(self):
        '''
        训练时，使用epsilon贪心策略随机或由神经网络（估值函数）做出决策，给出action（agent即将进行的动作）
        推断时，直接由神经网络做出决策
        0:沿当前方向前进
        1:左转
        2:右转
        '''
        # 随着训练过程进行，随机决策的概率应该逐步降低，epsilon下限设为50
        if(self.epsilon>50):
            self.epsilon-=self.current_epoch*0.01

        # 蛇当前行进的绝对方向
        cur_dir=int(self.game.snake.direction)
        sample=random.randint(0,500)
        # 随机选择actoin
        if(sample<self.epsilon):
            action=random.randint(0,2)

        # 神经网络选择action
        else:
            action=self.model.predict(self.get_state())
        
        
        #解析action（根据action计算新的绝对方向）
        if(action==0):
            return action,Direction(cur_dir)
        elif(action==1):
            return action,Direction((cur_dir-1)%4)
        elif(action==2):
            return action,Direction((cur_dir+1)%4)

    def get_state(self):
        '''
        从游戏中获取环境信息（状态）
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
        '''从经验池中获取一条或多条样本，成批量训练神经网络'''
        if len(self.memory) != 0:
            states, actions, rewards, next_states = zip(*(self.memory))
            self.model.train_step(states, actions, rewards, next_states)
    
    def train_short_memory(self, state, action, reward, next_state):
        '''训练单条样本，并将样本加入经验池，reward=-1的情况出现的较为频繁，少加点'''
        if reward == -1:
            if random.randint(1,15) == 1: 
                self.memory.append([state, action, reward, next_state])
        else:
            self.memory.append([state, action, reward, next_state])
        self.model.train_step([state], [action], [reward], [next_state])

    def train(self):
        '''整体训练过程'''
        while(True):
            # n轮保存一次模型
            if not self.current_epoch % self.n_save:
                    self.model.save()

            state=self.get_state()
            action=self.get_action()
            print(action)
            # 按action在真实环境中进行游玩，返回reward，游戏是否结束，和目前总分数
            reward,over,score=self.game.play_step(action[1])
            # 做完action后的下一步环境状态
            next_state=self.get_state()

            # 训练，train_long_memory的频率可以自己调节
            self.train_short_memory(state,action[0],reward,next_state)
            self.train_long_memory()
            # 显示信息
            print("======================================")
            print(f'epoch:{self.current_epoch},score:{score}\n')
            self.current_epoch+=1
            # 如果游戏结束，自动重开继续训练
            if(over):
                self.game.reset()
            # import pygame
            # pygame.time.delay(1000)

    def auto_play(self):
        '''ai自动游玩'''
        import pygame
        while(True):
            dir=self.get_action()[1]
            self.game.play_step(Direction(dir))
            # print(self.get_state())
            pygame.time.delay(150)

if __name__=='__main__':
    agent=Agent(Game(),DQN(),train=False)
    # agent.train()
    agent.auto_play()
    # # info=agent.game.get_env_info()

    # # print(info['food_pos'][0]-info['snake_body'][0][0])
    # # print(GRID_NUM)
    # # print(agent.get_state())
    # agent.auto_play()
    
    
