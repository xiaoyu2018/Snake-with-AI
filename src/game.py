import pygame
import sys
from food import *
from snake import *
from algorithms import Astar
from utils import get_game_config,get_ai_config
from pygame.locals import *


game_config=get_game_config()
#------------------------使用常量获取配置信息------------------------# 
GRID_SIZE = game_config['grid']['size']
LINE_WIDTH = game_config['grid']['line_width']
GRID_COLOR = game_config['grid']['color']
WINDOW_SIZE = game_config['window_size']
FONT_SIZE=game_config['font']['size']
FONT_COLOR=game_config['font']['color']
FONT_STYLE=game_config['font']['style']

class Game:
    def __init__(self) -> None:
    #------------------------绘制无需实时更新的部分------------------------# 
        pygame.init()
        pygame.display.set_caption('Snake')
        self.reset()

    def reset(self):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.screen.fill(game_config['bg_color'])
        for x in range(GRID_SIZE, WINDOW_SIZE[0],GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 2 * GRID_SIZE), (x, WINDOW_SIZE[1]), LINE_WIDTH)
        for y in range(2*GRID_SIZE, WINDOW_SIZE[1], GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y), LINE_WIDTH)
        self.game_over=False
        self.snake=Snake(self.screen)
        self.food=Food()
        self.score=0

    def control_by_keyboard(self,event):
        if event.type == KEYDOWN:
            if event.key in (K_w, K_UP):
                self.snake.move_up()
            elif event.key in (K_s, K_DOWN):
                self.snake.move_down()
            elif event.key in (K_a, K_LEFT):
                self.snake.move_left()
            elif event.key in (K_d, K_RIGHT):
                self.snake.move_right()
            

    def play_step(self,action:Direction=None):
        
        #------------------------实时绘制------------------------# 
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if not action and event.key in (K_w, K_UP,K_s, K_DOWN,K_a, K_LEFT,K_d, K_RIGHT):
                    self.control_by_keyboard(event)
                elif(self.game_over and event.key==K_r):
                    self.reset()
                    

        if(not self.game_over):
                
            self.food.update(self.screen)
            state=self.snake.update(self.food)
            
            # game over
            if(state==-1):
                self.game_over=True
                self.screen.blit(
                    pygame.font.SysFont(FONT_STYLE, 40).render(f"GAME OVER!", True, (200, 30, 30)),
                    ((WINDOW_SIZE[0]-200)>>1, 0)
                )
                return

            elif(state==1):
                self.score+=self.snake.length**2
                # 重绘顶部区域
                self.screen.fill(game_config['bg_color'], (0, 0, WINDOW_SIZE[0], 2 * GRID_SIZE))
                self.screen.blit(
                    pygame.font.SysFont(FONT_STYLE, FONT_SIZE).render(f"score: {self.score}", True, FONT_COLOR),
                    (5, 0)
                )
                
            
    
        pygame.display.update()
    
            

if __name__ == '__main__':
    game=Game()
    while True:
        game.play_step()
        pygame.time.delay(150)
