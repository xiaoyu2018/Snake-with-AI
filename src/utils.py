import json
from enum import Enum

def load_json(path:str):
    with open(path, 'r') as f:
        return json.load(f)

def get_game_config():
    return load_json('./config.json')['game']

def get_ai_config():
    return load_json('./config.json')['ai']


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3



game_config=get_game_config()
#------------------------使用常量获取配置信息------------------------# 
GRID_SIZE = game_config['grid']['size']
LINE_WIDTH = game_config['grid']['line_width']
GRID_COLOR = game_config['grid']['color']
WINDOW_SIZE = game_config['window_size']
FONT_SIZE=game_config['font']['size']
FONT_COLOR=game_config['font']['color']
FONT_STYLE=game_config['font']['style']
GRID_NUM=(WINDOW_SIZE[0]//GRID_SIZE,WINDOW_SIZE[1]//GRID_SIZE)

if __name__=='__main__':
    # print(GRID_NUM)
    ...