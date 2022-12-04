from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import argparse
import os

# def parse_args():
#     parse = argparse.ArgumentParser()  # 2、创建参数对象
#     parse.add_argument('--num_actors', type=int, default=18)  # 3、往参数对象添加参数
#     parse.add_argument('--episodes_per_actor', type=int, default=100000)  # 3、往参数对象添加参数
#     parse.add_argument('--batch_size', type=int, default=2048)
#     parse.add_argument('--load', action="store_false")
#     parse.add_argument('--load_model_dir', type=str,
#                        default='checkpoint/3_12288.pkl')
#     args = parse.parse_args()  # 4、解析参数对象获得解析对象
#     return args


if __name__ == '__main__':
    # args = parse_args()
    config = {
        'replay_buffer_size': 50000,
        'replay_buffer_episode': 400,
        'model_pool_size': 9,
        'actor_model_pool_name': 'actor_model-pool',
        'critic_model_pool_name': 'critic_model-pool',
        'num_actors': 24,
        'episodes_per_actor': 1000000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 2000,
        'batch_size': 1024,
        'epochs': 5,
        'clip': 0.2,
        'lr': 1e-4,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'value_coeff': 0.5,
        'entropy_coeff': 0.1,
        'device': 'cuda',
        'load': True,
        'load_model_dir': 'checkpoint/3_12288.pkl',
        'ckpt_save_interval': 1800,
        'ckpt_save_path': '/model/'
    }

    if not os.path.exists(config['ckpt_save_path']):
        print('mkdir '+ config['ckpt_save_path'])
        os.makedirs(config['ckpt_save_path'])
    else:
        print('ckpt_save_path already exists!')
    replay_buffer = ReplayBuffer(
        config['replay_buffer_size'], config['replay_buffer_episode'])

    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)

    for actor in actors:
        actor.start()
    learner.start()

    for actor in actors:
        actor.join()
    learner.terminate()
