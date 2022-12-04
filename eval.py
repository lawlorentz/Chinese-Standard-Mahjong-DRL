from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel


class eval():

    def __init__(self, config):
        self.config = config

    def run(self):
        models = []
        for i in range(4):
            model = CNNModel()
            model.train(False)  # Batch Norm inference mode
            models.append(model)

        # load initial model
        models[0].load_state_dict(torch.load(
            self.config['player1_model_dir'], map_location=torch.device('cpu')))
        models[1].load_state_dict(torch.load(
            self.config['player2_model_dir'], map_location=torch.device('cpu')))
        models[2].load_state_dict(torch.load(
            self.config['player3_model_dir'], map_location=torch.device('cpu')))
        models[3].load_state_dict(torch.load(
            self.config['player4_model_dir'], map_location=torch.device('cpu')))

        # collect data
        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
        policies = {player: models[i] for (i, player) in enumerate(
            env.agent_names)}  # all four players use the latest model
        all_rewards = {player: [] for player in env.agent_names}
        wins = {player: 0 for player in env.agent_names}

        for episode in range(self.config['episodes']):
            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state': {
                    'observation': [],
                    'action_mask': []
                },
                'action': [],
                'reward': [],
                'value': []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    model = policies[agent_name]
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(
                        state['observation'])
                    agent_data['state']['action_mask'].append(
                        state['action_mask'])
                    state['observation'] = torch.tensor(
                        state['observation'], dtype=torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(
                        state['action_mask'], dtype=torch.float).unsqueeze(0)

                    with torch.no_grad():
                        logits, value = model(state)
                        action_dist = torch.distributions.Categorical(
                            logits=logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(
                        rewards[agent_name])
                obs = next_obs
            print('Episode', episode, 'Reward', rewards)
            for agent_name in env.agent_names:
                all_rewards[agent_name].append(rewards[agent_name])
                if rewards[agent_name] > 0:
                    wins[agent_name] += 1
        return all_rewards,wins


if __name__ == '__main__':
    config = {
        'player1_model_dir': 'checkpoint/3_12288.pkl',
        'player2_model_dir': 'checkpoint/3_12288.pkl',
        'player3_model_dir': 'checkpoint/3_12288.pkl',
        'player4_model_dir': 'checkpoint/3_12288.pkl',
        'episodes': 1000,
    }
    e = eval(config)
    all_rewards,wins = e.run()
    print(wins)
