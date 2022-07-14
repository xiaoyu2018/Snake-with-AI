from game import Game
from algorithms.model import Model
import random
from collections import deque
from utils import Direction,GRID_NUM

class Agent:

    def __init__(self,game:Game,model:Model) -> None:
        self.game=game
        # 估值函数模型
        self.model=model
        # epsilon贪心策略的概率
        self.epsilon = 0
        # 记录当前游戏轮数
        self.current_epoch=0
        # 经验池(上限设为500)
        self.memory=deque(maxlen=500)

    def get_action(self,train=False):
        ...
    
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
        ...
    
    def train_short_memory(self):
        ...

    def auto_play(self):
        import pygame
        dir=random.randint(0,3)
        while(True):
            self.game.play_step(dir)
            print(self.get_state())
            pygame.time.delay(250)

if __name__=='__main__':
    agent=Agent(Game(),Model())
    # info=agent.game.get_env_info()

    # print(info['food_pos'][0]-info['snake_body'][0][0])
    # print(GRID_NUM)
    # print(agent.get_state())
    agent.auto_play()