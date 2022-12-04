from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
    
        # connect to model pool
        actor_model_pool = ModelPoolClient(self.config['actor_model_pool_name'])
        critic_model_pool = ModelPoolClient(self.config['critic_model_pool_name'])
        
        # create network model
        actor_model = CNNModel()
        critic_model = CNNModel()
        
        # load initial model
        actor_version = actor_model_pool.get_latest_model()
        actor_state_dict = actor_model_pool.load_model(actor_version)
        actor_model.load_state_dict(actor_state_dict)

        critic_version = critic_model_pool.get_latest_model()
        critic_state_dict = critic_model_pool.load_model(critic_version)
        critic_model.load_state_dict(critic_state_dict)
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        # policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        for episode in range(self.config['episodes_per_actor']):
            # update model
            actor_latest = actor_model_pool.get_latest_model()
            if actor_latest['id'] > actor_version['id']:
                actor_state_dict = actor_model_pool.load_model(actor_latest)
                actor_model.load_state_dict(actor_state_dict)
                actor_version = actor_latest
            
            critic_latest = critic_model_pool.get_latest_model()
            if critic_latest['id'] > critic_version['id']:
                critic_state_dict = critic_model_pool.load_model(critic_latest)
                critic_model.load_state_dict(critic_state_dict)
                critic_version = critic_latest
            
            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0)
                    actor_model.train(False) # Batch Norm inference mode
                    critic_model.train(False) # Batch Norm inference mode
                    with torch.no_grad():
                        logits = actor_model(state)[0]
                        value = critic_model(state)[1]
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs
            print(self.name, 'Episode', episode, 'Actor_Model', actor_latest['id'],'Critic_Model', critic_latest['id'], 'Reward', rewards)
            
            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                obs = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                actions = np.array(agent_data['action'], dtype = np.int64)
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                values = np.array(agent_data['value'], dtype = np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)
                
                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })