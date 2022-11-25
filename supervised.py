from dataset import MahjongGBDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import CNNModel
import torch.nn.functional as F
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
import argparse


def parse_args():
    parse = argparse.ArgumentParser()  # 2、创建参数对象
    parse.add_argument('--n_gpu', type=int, default=1)  # 3、往参数对象添加参数
    parse.add_argument('--logdir', type=str, default='/code/log/')
    parse.add_argument('--modeldir', type=str, default='/model')
    parse.add_argument('--load', action="store_true")
    parse.add_argument('--checkpoint', type=str, default='/model')
    args = parse.parse_args()  # 4、解析参数对象获得解析对象
    return args


log_writer = SummaryWriter()

def valid():
    print('Run validation:')
    correct = 0
    for _, d in enumerate(vloader):
        input_dict = {'is_training': False, 'obs': {
            'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
        with torch.no_grad():
            logits = model(input_dict)[0]
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, d[2].cuda()).sum().item()
    acc = correct / len(validateDataset)
    return acc

if __name__ == '__main__':
    args = parse_args()
    # 集群上跑用这个 logdir = '/code/log/'
    logdir = args.logdir
    n_gpu = args.n_gpu
    modeldir = args.modeldir
    load = args.load
    # resnet_depth=34
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    timestamp = int(time.time())
    os.mkdir(logdir + f'checkpoint_CNNModel_{timestamp}')

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024*n_gpu
    trainDataset = MahjongGBDataset(0, 0.9, True)
    loader = DataLoader(dataset=trainDataset,
                        batch_size=batchSize, shuffle=True)
    validateDataset = MahjongGBDataset(0.9, 1, False)
    vloader = DataLoader(dataset=validateDataset,
                         batch_size=batchSize, shuffle=False)

    # debug
    # for i in range(len(validateDataset)):
    #     a=validateDataset[i]
    #     print(i)

    # Load model
    model = CNNModel().to('cuda')
    if load:
        load_model_dir = args.checkpoint
        model.load_state_dict(torch.load(load_model_dir))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # Train and validate
    for e in range(50):
        print('Epoch', e)

        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {
                'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            logits = model(input_dict)[0]
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 128 == 0:
                print('Iteration %d/%d' % (i, len(trainDataset) //
                      batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1024 == 0 and not i == 0:
                print('Run validation:')
                correct = 0
                for _, d in enumerate(vloader):
                    input_dict = {'is_training': False, 'obs': {
                        'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
                    with torch.no_grad():
                        logits = model(input_dict)[0]
                        pred = logits.argmax(dim=1)
                        correct += torch.eq(pred, d[2].cuda()).sum().item()
                acc = correct / len(validateDataset)
                print(
                    f'Epoch {e + 1} {(i+1)/len(loader)*100}%, Validate acc: {acc}')
                scheduler.step(acc)
        # torch.save(model.module.state_dict(), logdir +
        #            f'checkpoint_CNNModel_{timestamp}/{e}.pkl')
        # print(logdir + f'checkpoint_CNNModel_{timestamp}/{e}.pkl saved')
            if i % 4096 == 0:
                model_path = modeldir + f'{e}_{i}.pkl'
                torch.save(model.module.state_dict(), model_path)
                print(model_path+' saved')

        model_path = modeldir + f'{e}.pkl'
        torch.save(model.module.state_dict(), model_path)
        print(model_path+' saved')

        # correct = 0
        # for i, d in enumerate(vloader):
        #     input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
        #     with torch.no_grad():
        #         logits = model(input_dict)
        #         pred = logits.argmax(dim = 1)
        #         correct += torch.eq(pred, d[2].cuda()).sum().item()
        # acc = correct / len(validateDataset)
        # print('Epoch', e + 1, 'Validate acc:', acc)
        # scheduler.step(acc)
        # log_writer.add_scalar('Accuracy/valid', float(acc), e)
