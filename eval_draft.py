from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel
models = []
for i in range(4):
    model=CNNModel()
    models.append(model)
        
# load initial model
models[0].load_state_dict(torch.load('checkpoint/3_12288.pkl',map_location=torch.device('cpu')))
models[1].load_state_dict(torch.load('checkpoint/3_12288.pkl',map_location=torch.device('cpu')))
models[2].load_state_dict(torch.load('checkpoint/3_12288.pkl',map_location=torch.device('cpu')))
models[3].load_state_dict(torch.load('checkpoint/3_12288.pkl',map_location=torch.device('cpu')))

# collect data
env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
policies = {player : models[i] for (i,player) in enumerate(env.agent_names)} # all four players use the latest model
