import pygame
from utils import get_game_config
from enum import Enum
from collections import deque
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
        self.screen=screen
        self.length=3

        # 存储蛇身体的格子坐标，设置初始值  
        self.body=deque([
            (init_pos[0],init_pos[1]),
            (init_pos[0]-1,init_pos[1]),
            (init_pos[0]-2,init_pos[1]),
            ])
        # 初始化贪吃蛇时，绘制整个蛇身
        for pos in self.body:
            self.fill(screen,pos,self.color)

        
    
    def move_up(self):
        if(self.direction!=Direction.DOWN):
            self.direction=Direction.UP
    def move_down(self):
        if(self.direction!=Direction.UP):
            self.direction=Direction.DOWN
    def move_left(self):
        if(self.direction!=Direction.RIGHT):
            self.direction=Direction.LEFT
    def move_right(self):
        if(self.direction!=Direction.LEFT):
            self.direction=Direction.RIGHT


    def fill(self,screen,grid,color):
        # 绘制蛇身体
        pygame.draw.rect(
                screen,color,
                (
                    *(i*self.size+self.line_width for i in grid),
                    *(self.size-2*self.line_width,)*2
                ),
                0
            )
        
        
    def update(self,food):
        # 更新蛇身列表，并重新绘制蛇头和蛇尾格子
        new_head=self.body[0]
        
        if(self.direction==Direction.UP):
            new_head=(new_head[0],new_head[1]-1)
        elif(self.direction==Direction.DOWN):
            new_head=(new_head[0],new_head[1]+1)
        elif(self.direction==Direction.LEFT):
            new_head=(new_head[0]-1,new_head[1])
        elif(self.direction==Direction.RIGHT):
            new_head=(new_head[0]+1,new_head[1])
        # print(new_head)

        # 绘制前判断是否撞墙或撞自己，如果撞墙或撞自己，则游戏结束
        if(
            (new_head in self.body) or 
            new_head[0]<0 or (new_head[0])*self.size>=self.win_size[0] or
            new_head[1]<2 or (new_head[1])*self.size>=self.win_size[1]
            ):
                return -1
        self.fill(self.screen,new_head,self.color)
        self.body.appendleft(new_head)
        
        # 判断蛇是否吃到食物
        if(self.body[0]!=(food.pos_x,food.pos_y)):
            self.fill(self.screen,self.body[-1],game_config['bg_color'])
            self.body.pop()
            return 0
        
        self.length+=1
        food.gen_pos()
        while(food.pos_x,food.pos_y) in self.body:
            food.gen_pos()
        return 1
            


if __name__=='__main__':
    a=deque([1,1,3])
    print(a[0] in list(a)[1:])