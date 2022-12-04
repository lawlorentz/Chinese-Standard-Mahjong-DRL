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
        actor_model_pool = ModelPoolServer(
            self.config['model_pool_size'], self.config['actor_model_pool_name'])
        critic_model_pool = ModelPoolServer(
            self.config['model_pool_size'], self.config['critic_model_pool_name'])

        # initialize model params
        device = torch.device(self.config['device'])
        actor_model = CNNModel()
        critic_model = CNNModel()

        if self.config['load']:
            actor_model.load_state_dict(torch.load(self.config['load_model_dir']))
            critic_model.load_state_dict(torch.load(self.config['load_model_dir']))

        # send to model pool
        # push cpu-only tensor to model_pool
        actor_model_pool.push(actor_model.state_dict())
        actor_model = actor_model.to(device)

        critic_model_pool.push(critic_model.state_dict())
        critic_model = critic_model.to(device)

        # training
        actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=self.config['actor_lr'])
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=self.config['critic_lr'])

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
            actor_model.train(True)  # Batch Norm training mode
            critic_model.train(True)  # Batch Norm training mode

            old_logits = actor_model(states)[0]
            old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
            old_log_probs = torch.log(old_probs+1e-8).detach()
            for _ in range(self.config['epochs']):
                logits, values = actor_model(states)[0],critic_model(states)[1]
                action_dist = torch.distributions.Categorical(logits=logits)
                probs = F.softmax(logits, dim=1).gather(1, actions)
                log_probs = torch.log(probs+1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(
                    ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                # loss = policy_loss + \
                #     self.config['value_coeff'] * value_loss + \
                #     self.config['entropy_coeff'] * entropy_loss
                # print(f'{_}: origin  :',policy_loss.item(), value_loss.item(), entropy_loss.item())
                # print(f'{_}: weighted:',policy_loss.item(), self.config['value_coeff'] *value_loss.item(), self.config['entropy_coeff']*entropy_loss.item(), loss.item())
                print(f'{_}: origin  :',policy_loss.item(), value_loss.item(), entropy_loss.item())
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                # loss.backward()
                policy_loss.backward()
                value_loss.backward()
                actor_optimizer.step()
                critic_optimizer.step()

            # push new model
            actor_model = actor_model.to('cpu')
            critic_model = critic_model.to('cpu')
            # push cpu-only tensor to model_pool
            actor_model_pool.push(actor_model.state_dict())
            actor_model = actor_model.to(device)
            critic_model_pool.push(critic_model.state_dict())
            critic_model = critic_model.to(device)

            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + \
                    'actor_model_%d.pt' % iterations
                torch.save(actor_model.state_dict(), path)
                print(f'saving {path}')
                path = self.config['ckpt_save_path'] + \
                    'critic_model_%d.pt' % iterations
                torch.save(critic_model.state_dict(), path)
                print(f'saving {path}')
                cur_time = t
            iterations += 1
            self.replay_buffer.clear()