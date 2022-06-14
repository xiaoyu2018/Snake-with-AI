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

    while(True):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        
        screen.blit(
            pygame.font.SysFont(FONT_STYLE, FONT_SIZE).render(f"score: {score}", True, FONT_COLOR),
            (5, 0)
        )

        if(game_over):
            screen.blit(
                pygame.font.SysFont(FONT_STYLE, 40).render(f"GAME OVER!", True, (200, 30, 30)),
                ((WINDOW_SIZE[0]-200)>>1, 0)
            )
        
        food=Food()
        # food.gen_pos()
        food.fill(screen)
        # food.clean(screen)
        

        snake=Snake(screen)
        # snake.draw(screen)

        
        
        
        
        
        pygame.display.update()


main()