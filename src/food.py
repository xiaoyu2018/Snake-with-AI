import random
import pygame
from utils import get_game_config

game_config=get_game_config()
class Food:

    def __init__(
        self,
        pos_x=get_game_config()['window_size'][0]//2,
        pos_y=get_game_config()['window_size'][1]//2
        ) -> None:
        self.size=get_game_config()['grid']['size']
        self.win_size=get_game_config()['window_size']
        self.line_width=get_game_config()['grid']['line_width']
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.color=(230, 210, 111)
        
    def gen_pos(self):
        self.pos_x=random.randint(0,self.win_size[0]//self.size-1)*self.size
        self.pos_y=random.randint(2,self.win_size[1]//self.size-1)*self.size
    
    def update(self,screen):
        pygame.draw.rect(
            screen,self.color,
            (
                self.pos_x+self.line_width,self.pos_y+self.line_width,
                self.size-2*self.line_width,self.size-2*self.line_width
                ),
            0
        )
    
    def clean(self,screen):
        pygame.draw.rect(
            screen,game_config['bg_color'],
            (
                self.pos_x+self.line_width,self.pos_y+self.line_width,
                self.size-2*self.line_width,self.size-2*self.line_width
                ),
            0
        )
    