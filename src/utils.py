import json

def load_json(path:str):
    with open(path, 'r') as f:
        return json.load(f)

def get_game_config():
    return load_json('./config.json')['game']

def get_ai_config():
    return load_json('./config.json')['ai']
