from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
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
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)
            
            sample_in = self.replay_buffer.stats['sample_in']
            sample_out = self.replay_buffer.stats['sample_out']
            print(
                f'Iteration {iterations}, replay buffer in {sample_in} out {sample_out}, ratio: {sample_out/sample_in}')            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8).detach()
            avg_policy_loss, avg_value_loss, avg_entropy_loss = 0, 0, 0
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                all_probs = F.softmax(logits, dim=1)
                action_dist = torch.distributions.Categorical(probs = all_probs)

                probs = all_probs.gather(1, actions)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss

                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_policy_loss /= self.config['epochs']
            avg_value_loss /= self.config['epochs']
            avg_entropy_loss /= self.config['epochs']
            print(f'Iter: {iterations}, avg_policy_loss: {avg_policy_loss}, avg_value_loss: {avg_value_loss}, avg_entropy_loss: {avg_entropy_loss}')
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