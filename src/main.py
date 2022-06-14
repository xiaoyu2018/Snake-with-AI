import pygame
import sys
from food import *
from snake import *
from algorithms import Astar
from utils import get_game_config,get_ai_config
from pygame.locals import *




def main():
    game_config=get_game_config()
#------------------------使用常量获取配置信息------------------------# 
    GRID_SIZE = game_config['grid']['size']
    LINE_WIDTH = game_config['grid']['line_width']
    GRID_COLOR = game_config['grid']['color']
    WINDOW_SIZE = game_config['window_size']
    FONT_SIZE=game_config['font']['size']
    FONT_COLOR=game_config['font']['color']
    FONT_STYLE=game_config['font']['style']

#------------------------绘制无需实时更新的部分------------------------# 
    pygame.init()
    pygame.display.set_caption('Snake')
    screen = pygame.display.set_mode(WINDOW_SIZE)
    screen.fill(game_config['bg_color'])
    
    # 画网格线（最上方留出计分区域）
    for x in range(GRID_SIZE, WINDOW_SIZE[0],GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 2 * GRID_SIZE), (x, WINDOW_SIZE[1]), LINE_WIDTH)
    for y in range(2*GRID_SIZE, WINDOW_SIZE[1], GRID_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y), LINE_WIDTH)


#------------------------实时绘制------------------------# 
    game_over = False
    score=0
    delay=1000
    snake=Snake(screen)
    food=Food()

    while(True):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key in (K_w, K_UP):
                    snake.move_up()
                elif event.key in (K_s, K_DOWN):
                    snake.move_down()
                elif event.key in (K_a, K_LEFT):
                    snake.move_left()
                elif event.key in (K_d, K_RIGHT):
                    snake.move_right()
                elif(game_over and event.key==K_r):
                    
                    screen.fill(game_config['bg_color'])
                    for x in range(GRID_SIZE, WINDOW_SIZE[0],GRID_SIZE):
                        pygame.draw.line(screen, GRID_COLOR, (x, 2 * GRID_SIZE), (x, WINDOW_SIZE[1]), LINE_WIDTH)
                    for y in range(2*GRID_SIZE, WINDOW_SIZE[1], GRID_SIZE):
                        pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y), LINE_WIDTH)
                    game_over=False
                    snake=Snake(screen)
                    food=Food()
                    score=0
                    delay=1000


        if(delay == 1000 and not game_over):
            
            if(
                snake.body[0] in snake.body[1:] or 
                snake.body[0][0]<=0 or (snake.body[0][0]+1)*GRID_SIZE>=WINDOW_SIZE[0] or
                snake.body[0][1]<=0 or (snake.body[0][1]+1)*GRID_SIZE>=WINDOW_SIZE[1]
            ):
                game_over=True
                
    

            if(game_over):
                screen.blit(
                    pygame.font.SysFont(FONT_STYLE, 40).render(f"GAME OVER!", True, (200, 30, 30)),
                    ((WINDOW_SIZE[0]-200)>>1, 0)
            )
                continue
            
            
            
            food.update(screen)
            if(snake.update(food)):
                score+=20
                # 重绘顶部区域
                screen.fill(game_config['bg_color'], (0, 0, WINDOW_SIZE[0], 2 * GRID_SIZE))
                screen.blit(
                    pygame.font.SysFont(FONT_STYLE, FONT_SIZE).render(f"score: {score}", True, FONT_COLOR),
                    (5, 0)
                )
        
        delay-=0.5
        delay=1000 if not delay else delay
        
        pygame.display.update()


main()