import pygame
from utils import get_game_config
from enum import Enum

game_config=get_game_config()


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Snake:

    def __init__(self,screen) -> None:
        self.size=get_game_config()['grid']['size']
        self.win_size=get_game_config()['window_size']
        self.line_width=get_game_config()['grid']['line_width']
        grid_x_count=self.win_size[0]//self.size
        grid_y_count=self.win_size[1]//self.size
        init_pos=(grid_x_count//2-2,grid_y_count//2)
        self.color=(100, 100, 100)
        self.direction=Direction.RIGHT
        # 存储蛇身体的格子坐标，设置初始值  
        self.body=[
            (init_pos[0],init_pos[1]),
            (init_pos[0]-1,init_pos[1]),
            (init_pos[0]-2,init_pos[1]),
            ]
        # 初始化贪吃蛇时，绘制整个蛇身
        for pos in self.body:
            self.fill(screen,pos)

        
    
    def move_up(self):
        self.direction=Direction.UP
    def move_down(self):
        self.direction=Direction.DOWN
    def move_left(self):
        self.direction=Direction.LEFT
    def move_right(self):
        self.direction=Direction.RIGHT


    def fill(self,screen,grid):
        # 绘制蛇身体
        pygame.draw.rect(
                screen,self.color,
                (
                    *(i*self.size+self.line_width for i in grid),
                    *(self.size-2*self.line_width,)*2
                ),
                0
            )
    def eat(self):
        # 吃到食物，蛇身体增加一格,直接绘制
        new_tail=self.body[-1]
        if(self.direction==Direction.UP):
            new_tail=(new_tail[0],new_tail[1]+1)
        elif(self.direction==Direction.DOWN):
            new_tail=(new_tail[0],new_tail[1]-1)
        elif(self.direction==Direction.LEFT):
            new_tail=(new_tail[0]+1,new_tail[1])
        elif(self.direction==Direction.RIGHT):
            new_tail=(new_tail[0]-1,new_tail[1])
        self.body.append(new_tail)
        
    def update(self,screen):
        # 更新蛇身列表，并重新绘制蛇头和蛇尾格子
        ... 
        