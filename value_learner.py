from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class ValueLearner(Process):
    
    def __init__(self, config, replay_buffer):
        super(ValueLearner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
    
    def run(self):
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()
        if self.config['load']:
            model.load_state_dict(torch.load(self.config['load_model_dir']))
        
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        # Freeze parameters, only left value branch
        for name, parameter in model.named_parameters():
            if '_value_branch' not in name:
                parameter.requires_grad = False
        print(model.state_dict)

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        while True:
            while self.replay_buffer.size() < self.config['min_sample']:
                time.sleep(0.1)
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            targets = torch.tensor(batch['target']).to(device)
            
            sample_in = self.replay_buffer.stats['sample_in']
            sample_out = self.replay_buffer.stats['sample_out']
            print(
                f'Iteration {iterations}, replay buffer in {sample_in} out {sample_out}, ratio: {sample_out/sample_in}')            

            sum_loss = 0.0
            model.train(True) # Batch Norm training mode
            for epo_id in range(self.config['epochs']):
                logits, values = model(states)
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                loss = value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(f"Iter: {iterations}, epoch: {epo_id}, loss: {loss.item()}")
                sum_loss += loss.item()
            print(f"Iter: {iterations} avg_loss: {sum_loss / self.config['epochs']}")

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                torch.save(model.state_dict(), path)
                cur_time = t
            iterations += 1