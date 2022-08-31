
'''
向agent提供决策算法的统一接口
具体算法如dqn负责实现此接口
'''

class Model:
    def __init__(self):
        pass

    def train_step(self):
        pass

    def predict(self):
        pass
    
    def save(self):
        pass

    def load(self,param_path):
        pass