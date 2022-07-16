import pygame
import sys
from food import *
from snake import *
from utils import *
from pygame.locals import *




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
        

    def get_env_info(self):
        info=dict()
        info['snake_direction']=self.snake.direction
        info['snake_body']=self.snake.body
        info['food_pos']=(self.food.pos_x,self.food.pos_y)
        info['border']=WINDOW_SIZE[0]//GRID_SIZE,WINDOW_SIZE[1]//GRID_SIZE
        return info
        
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
        
        # 没吃到食物，但也没有游戏结束
        reward=0
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
                            
        if(action!=None):
            if(action==Direction.UP):
                self.snake.move_up()
            elif(action==Direction.DOWN):
                self.snake.move_down()
            elif(action==Direction.LEFT):
                self.snake.move_left()
            elif(action==Direction.RIGHT):
                self.snake.move_right()

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
                reward=-10

            # 吃到食物
            elif(state==1):
                self.score+=self.snake.length**2
                # 重绘顶部区域
                self.screen.fill(game_config['bg_color'], (0, 0, WINDOW_SIZE[0], 2 * GRID_SIZE))
                self.screen.blit(
                    pygame.font.SysFont(FONT_STYLE, FONT_SIZE).render(f"score: {self.score}", True, FONT_COLOR),
                    (5, 0)
                )
                reward=15
                
        pygame.display.update()
        return reward,self.game_over,self.score
        
    
            

if __name__ == '__main__':
    game=Game()
    while True:
        print(game.play_step())
        print(game.get_env_info())
        pygame.time.delay(150)
