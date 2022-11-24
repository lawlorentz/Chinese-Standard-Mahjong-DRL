# 饼万条互换
import numpy as np
import json

feature_num=70

with open('data/count.json') as f:
    match_samples = json.load(f)
total_matches = len(match_samples)


obs_augment_index = [[1, 2, 3], [1, 3, 2], [
    2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]


def action2response(action):
    if action < OFFSET_ACT['Hu']:
        return 'Pass'
    if action < OFFSET_ACT['Play']:
        return 'Hu'
    if action < OFFSET_ACT['Chi']:
        return 'Play ' + TILE_LIST[action - OFFSET_ACT['Play']]
    if action < OFFSET_ACT['Peng']:
        t = (action - OFFSET_ACT['Chi']) // 3
        return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
    if action < OFFSET_ACT['Gang']:
        return 'Peng'
    if action < OFFSET_ACT['AnGang']:
        return 'Gang'
    if action < OFFSET_ACT['BuGang']:
        return 'Gang ' + TILE_LIST[action - OFFSET_ACT['AnGang']]
    return 'BuGang ' + TILE_LIST[action - OFFSET_ACT['BuGang']]


def response2action(self, response):
    t = response.split()
    if t[0] == 'Pass':
        return OFFSET_ACT['Pass']
    if t[0] == 'Hu':
        return OFFSET_ACT['Hu']
    if t[0] == 'Play':
        return OFFSET_ACT['Play'] + OFFSET_TILE[t[1]]
    if t[0] == 'Chi':
        return OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
    if t[0] == 'Peng':
        return OFFSET_ACT['Peng'] + OFFSET_TILE[t[1]]
    if t[0] == 'Gang':
        return OFFSET_ACT['Gang'] + OFFSET_TILE[t[1]]
    if t[0] == 'AnGang':
        return OFFSET_ACT['AnGang'] + OFFSET_TILE[t[1]]
    if t[0] == 'BuGang':
        return OFFSET_ACT['BuGang'] + OFFSET_TILE[t[1]]
    return OFFSET_ACT['Pass']


OFFSET_ACT = {

    'Pass': 0,
    'Hu': 1,
    'Play': 2,
    'Chi': 36,
    'Peng': 99,
    'Gang': 133,
    'AnGang': 167,
    'BuGang': 201
}

TILE_LIST = [
    *('W%d' % (i+1) for i in range(9)),
    *('T%d' % (i+1) for i in range(9)),
    *('B%d' % (i+1) for i in range(9)),
    *('F%d' % (i+1) for i in range(4)),
    *('J%d' % (i+1) for i in range(3))
]
OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}


OFFSET_ACT1 = {
    'Pass': 0,
    'Hu': 1,
    'Play_W': 2,
    'Play_T': 11,
    'Play_B': 20,
    'Chi_W': 36,
    'Chi_T': 57,
    'Chi_B': 78,
    'Peng': 99,
    'Gang': 133,
    'AnGang': 167,
    'BuGang': 201
}


def data_augment(index):
    d = np.load('data/cooked_data_without0/%d.npz' % index)
    cache = {'obs': d['obs'], 'mask': d['mask'], 'act': d['act']}
    for k in d:
        cache[k] = d[k]

    obs_augment = []
    mask_augment = []
    act_augment = []

    obs_1 = cache['obs'].copy()
    act_1 = np.zeros([cache['act'].size, 235])
    for i in range(cache['act'].size):
        act_1[i][cache['act'][i]] = 1
    act_0 = act_1.copy()
    mask_1 = cache['mask'].copy()

    for i in range(6):
        # obs
        obs_1 = cache['obs'].copy()
        for j in range(3):
            obs_1[:, 2:feature_num, j, :] = cache['obs'][:,
                                                 2:feature_num, obs_augment_index[i][j]-1, :]

        obs_augment.append(obs_1)

        # mask
        mask_1 = cache['mask'].copy()
        for j in range(3):
            mask_1[:, OFFSET_ACT1['Chi_W']+21*j:OFFSET_ACT1['Chi_T']+21*j] = cache['mask'][:, OFFSET_ACT1['Chi_W'] +
                                                                                           21*(obs_augment_index[i][j]-1):OFFSET_ACT1['Chi_T']+21*(obs_augment_index[i][j]-1)]
            mask_1[:, OFFSET_ACT1['Play_W']+9*j:OFFSET_ACT1['Play_T']+9*j] = cache['mask'][:, OFFSET_ACT1['Play_W'] +
                                                                                           9*(obs_augment_index[i][j]-1):OFFSET_ACT1['Play_T']+9*(obs_augment_index[i][j]-1)]
            mask_1[:, OFFSET_ACT1['Peng']+9*j:OFFSET_ACT1['Peng']+9*(j+1)] = cache['mask'][:, OFFSET_ACT1['Peng'] +
                                                                                           9*(obs_augment_index[i][j]-1):OFFSET_ACT1['Peng']+9*(obs_augment_index[i][j])]
            mask_1[:, OFFSET_ACT1['Gang']+9*j:OFFSET_ACT1['Gang']+9*(j+1)] = cache['mask'][:, OFFSET_ACT1['Gang'] +
                                                                                           9*(obs_augment_index[i][j]-1):OFFSET_ACT1['Gang']+9*(obs_augment_index[i][j])]
            mask_1[:, OFFSET_ACT1['AnGang']+9*j:OFFSET_ACT1['AnGang']+9*(j+1)] = cache['mask'][:, OFFSET_ACT1['AnGang'] +
                                                                                               9*(obs_augment_index[i][j]-1):OFFSET_ACT1['AnGang']+9*(obs_augment_index[i][j])]
            mask_1[:, OFFSET_ACT1['BuGang']+9*j:OFFSET_ACT1['BuGang']+9*(j+1)] = cache['mask'][:, OFFSET_ACT1['BuGang'] + 9*(
                obs_augment_index[i][j]-1):OFFSET_ACT1['BuGang']+9*(obs_augment_index[i][j])]
        mask_augment.append(mask_1)

        # act
        act_1 = np.zeros([cache['act'].size, 235])
        for i_ in range(cache['act'].size):
            act_1[i_][cache['act'][i_]] = 1
        for j in range(3):
            # print(act_1[:, OFFSET_ACT['Chi_W']+21*j:OFFSET_ACT['Chi_T']+21*j])
            # print(OFFSET_ACT['Chi_W'] +21*(obs_augment_index[i][j]-1))
            # print(OFFSET_ACT['Chi_T']+21*(obs_augment_index[i][j]-1))
            # print(act_0[:, OFFSET_ACT['Chi_W'] +21*(obs_augment_index[i][j]-1):OFFSET_ACT['Chi_T']+21*(obs_augment_index[i][j]-1)])
            act_1[:, OFFSET_ACT1['Chi_W']+21*j:OFFSET_ACT1['Chi_T']+21*j] = act_0[:, OFFSET_ACT1['Chi_W'] +
                                                                                  21*(obs_augment_index[i][j]-1):OFFSET_ACT1['Chi_T']+21*(obs_augment_index[i][j]-1)]
            act_1[:, OFFSET_ACT1['Play_W']+9*j:OFFSET_ACT1['Play_T']+9*j] = act_0[:, OFFSET_ACT1['Play_W'] +
                                                                                  9*(obs_augment_index[i][j]-1):OFFSET_ACT1['Play_T']+9*(obs_augment_index[i][j]-1)]
            act_1[:, OFFSET_ACT1['Peng']+9*j:OFFSET_ACT1['Peng']+9*(j+1)] = act_0[:, OFFSET_ACT1['Peng'] +
                                                                                  9*(obs_augment_index[i][j]-1):OFFSET_ACT1['Peng']+9*(obs_augment_index[i][j])]
            act_1[:, OFFSET_ACT1['Gang']+9*j:OFFSET_ACT1['Gang']+9*(j+1)] = act_0[:, OFFSET_ACT1['Gang'] +
                                                                                  9*(obs_augment_index[i][j]-1):OFFSET_ACT1['Gang']+9*(obs_augment_index[i][j])]
            act_1[:, OFFSET_ACT1['AnGang']+9*j:OFFSET_ACT1['AnGang']+9*(j+1)] = act_0[:, OFFSET_ACT1['AnGang'] +
                                                                                      9*(obs_augment_index[i][j]-1):OFFSET_ACT1['AnGang']+9*(obs_augment_index[i][j])]
            act_1[:, OFFSET_ACT1['BuGang']+9*j:OFFSET_ACT1['BuGang']+9*(j+1)] = act_0[:, OFFSET_ACT1['BuGang'] + 9*(
                obs_augment_index[i][j]-1):OFFSET_ACT1['BuGang']+9*(obs_augment_index[i][j])]
        act_augment.append(act_1)

    obs_ = np.concatenate(tuple(obs_augment), axis=0)
    mask_ = np.concatenate(tuple(mask_augment), axis=0)
    act_one_hot = np.concatenate(tuple(act_augment), axis=0)
    act_ = np.zeros([act_one_hot.shape[0]]).astype(np.int32)
    for i in range(act_one_hot.shape[0]):
        act_[i] = np.argmax(act_one_hot[i])

    np.savez('data/cooked_data_without0/%d_augmented_%d.npz' % (index,feature_num),
             obs=obs_,
             mask=mask_,
             act=act_)
    if index % 1024 == 0:
        print('data %d augmented and saved' % index)


for i in range(total_matches):
    data_augment(i)
