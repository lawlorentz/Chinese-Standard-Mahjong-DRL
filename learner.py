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
        model_pool = ModelPoolServer(
            self.config['model_pool_size'], self.config['model_pool_name'])

        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel().to(device)

        if self.config['load']:
            model.load_state_dict(torch.load(self.config['load_model_dir']))

        # send to model pool
        model_pool.push(model.state_dict())

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])

        save_freq = 1000
        n = 0
        while True:
            n += 1
            # wait for initial samples
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

            print('Replay buffer in %d out %d' % (
                self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))

            # calculate PPO loss
            model.train(True)  # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
            old_log_probs = torch.log(old_probs).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                action_dist = torch.distributions.Categorical(logits=logits)
                probs = F.softmax(logits, dim=1).gather(1, actions)
                log_probs = torch.log(probs)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(
                    ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(
                    values, targets.unsqueeze(-1)))
                entropy_loss = -torch.mean(action_dist.entropy())
                print(f'{_}: origin  :', policy_loss.item(),
                      value_loss.item(), entropy_loss.item())
                print(f'{_}: weighted:', policy_loss.item(
                ), self.config['value_coeff'] * value_loss.item(), self.config['entropy_coeff']*entropy_loss.item())
                loss = policy_loss + \
                    self.config['value_coeff'] * value_loss + \
                    self.config['entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # push new model
            model_pool.push(model.state_dict())
            if n % save_freq == 0:
                torch.save(model.state_dict(), '/model/1234.pkl')
