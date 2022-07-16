
'''
向agent提供估值函数的统一接口
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