import numpy as np
import torch
import torch.nn.functional as F

from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel


def run(models: list, device, total_episode=100, record_data=False):
    env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
    policies = {player : model for player, model in zip(env.agent_names, models)} # all four players use the latest model

    if record_data:
        datas = []
    
    win_num = {player : 0 for player in env.agent_names}
    total_rewards = {player : 0 for player in env.agent_names}

    for episode in range(total_episode):
        # run one episode
        obs = env.reset()

        if record_data:
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'logits' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
        
        done = False
        winner = "None    "
        while not done:
            # each player take action
            actions = {}
            values = {}
            for agent_name in obs:
                if record_data:
                    agent_data = episode_data[agent_name]
                    # agent_data['state']['observation'].append(state['observation'])
                    # agent_data['state']['action_mask'].append(state['action_mask'])
                
                state = obs[agent_name]
                state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0).to(device)
                state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0).to(device)
                
                policies[agent_name].train(False) # Batch Norm inference mode
                with torch.no_grad():
                    logits, value = policies[agent_name](state)
                    soft_log = F.softmax(logits, dim=-1)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    entropy = torch.mean(action_dist.entropy())
                    action = action_dist.sample().item()
                    value = value.item()
                actions[agent_name] = action
                values[agent_name] = value
                
                if record_data:
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                    if agent_name == 'player_1':
                        print(entropy)
            # interact with env
            next_obs, rewards, done = env.step(actions)
            if done:
                for key in rewards.keys():
                    if rewards[key] > 0:
                        winner = key
                        break
                if winner != "None    ":
                    win_num[winner] += 1
                for key in rewards.keys():
                    total_rewards[key] += rewards[key]

            if record_data:
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
            
            obs = next_obs
        print('Episode', episode, 'Winner', winner, 'Reward', rewards)
        if record_data:
            datas.append(episode_data)
    
    win_rate = {player : f"{100.0 * win_num[player] / total_episode:.2f}%" for player in env.agent_names}
    print("Results:")
    print(f"Total episode: {total_episode}")
    print("win:", win_num)
    print("rate:", win_rate)
    print("reward:", total_rewards)
    
    if record_data:
        import json
        import time
        import os
        time_stamp = time.time()
        time_array = time.localtime(time_stamp)
        str_time_stamp = time.strftime("%Y%m%d%H%M%S", time_array)
        if not os.path.exists("battle_records"):
            os.mkdir("battle_records")
        str_json = json.dumps(datas, indent=2)
        with open(f"./battle_records/{str_time_stamp}.json", "w") as f:
            f.write(str_json)


def main():
    # device = "cuda"
    device = "cpu"
    models = [CNNModel(), CNNModel(), CNNModel(), CNNModel()]
    models[0].load_state_dict(torch.load("checkpoint/3_12288.pkl",map_location=device))
    models[1].load_state_dict(torch.load("checkpoint/3_12288.pkl",map_location=device))
    models[2].load_state_dict(torch.load("checkpoint/model_78789.pt",map_location=device))
    models[3].load_state_dict(torch.load("checkpoint/model_78789.pt",map_location=device))
    for i in range(4):
        models[i].to(device)

    run(models, device, total_episode=1, record_data=True)

if __name__ == "__main__":
    main()