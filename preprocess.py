from feature import FeatureAgent
import numpy as np
import json

obs = [[] for i in range(4)]
actions = [[] for i in range(4)]
matchid = -1

l = []

def saveData():
    # 清洗数据
    global actions, obs
    useful_ids = [[i for i, a in enumerate(actions[j]) if np.sum(obs[j][i]['action_mask']) != 1] for j in range(4)]
    actions = [[actions[j][i] for i in useful_ids[j]] for j in range(4)]
    obs = [[obs[j][i] for i in useful_ids[j]] for j in range(4)]
    assert [len(x) for x in obs] == [len(x) for x in actions], 'obs actions not matching!'
    l.append(sum([len(x) for x in obs]))
    np.savez('data/cooked_data_without0/%d.npz'%matchid
        , obs = np.stack([x['observation'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , mask = np.stack([x['action_mask'] for i in range(4) for x in obs[i]]).astype(np.int8)
        , act = np.array([x for i in range(4) for x in actions[i]])
    )
    for x in obs: x.clear()
    for x in actions: x.clear()

with open('data/data.txt', encoding='UTF-8') as f:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == 'Match':
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
            print('Processing match %d %s...' % (matchid, t[1]))
        elif t[0] == 'Wind':
            for agent in agents:
                agent.request2obs(line)
        elif t[0] == 'Player':
            p = int(t[1])
            if t[2] == 'Deal':
                agents[p].request2obs(' '.join(t[2:]))
            # 摸牌
            elif t[2] == 'Draw':
                for i in range(4):
                    if i == p:
                        # 对于摸牌的人，传入 Draw xxx
                        obs[p].append(agents[p].request2obs(' '.join(t[2:])))
                        actions[p].append(0)
                    else:
                        # 对于没有摸牌的人，传入 xxx Draw
                        agents[i].request2obs(' '.join(t[:3]))
            # 打牌
            elif t[2] == 'Play':
                actions[p].pop()
                actions[p].append(agents[p].response2action(' '.join(t[2:])))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs(line)
                    else:
                        obs[i].append(agents[i].request2obs(line))
                        actions[i].append(0)
                # 被打出来的这张牌
                curTile = t[3]
            elif t[2] == 'Chi':
                actions[p].pop()
                # curTile是被吃掉的牌，t[3]是顺子中间的牌
                actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3])))
                        # p吃了别人的牌
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
            elif t[2] == 'Peng':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                for i in range(4):
                    if i == p:
                        obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3])))
                        actions[p].append(0)
                    else:
                        agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
            elif t[2] == 'Gang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                for i in range(4):
                    agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
            elif t[2] == 'AnGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                    else:
                        agents[i].request2obs('Player %d AnGang' % p)
            elif t[2] == 'BuGang':
                actions[p].pop()
                actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                for i in range(4):
                    if i == p:
                        agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                    else:
                        obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3])))
                        actions[i].append(0)
            elif t[2] == 'Hu':
                actions[p].pop()
                actions[p].append(agents[p].response2action('Hu'))
            # 处理抢碰杠胡的情况
            if t[2] in ['Peng', 'Gang', 'Hu']:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == 'Chi':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                        elif t[k + 2] == 'Peng':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                        elif t[k + 2] == 'Gang':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                        elif t[k + 2] == 'Hu':
                            actions[p].pop()
                            actions[p].append(agents[p].response2action('Hu'))
                    else: break
        elif t[0] == 'Score':
            saveData()
        line = f.readline()
with open('data/count.json', 'w') as f:
    json.dump(l, f)