# 基于强化学习的国标麻将AI

## 配置

需要安装算番库[PyMahjongGB](https://github.com/ailab-pku/PyMahjongGB)

## 介绍

`feature.py`定义71维feature：

```python
OFFSET_OBS = {
    # 门风
    'SEAT_WIND' : 0,
    # 圈风
    'PREVALENT_WIND' : 1,
    # 自己的手牌
    'HAND' : 2,
    # 每个位置的弃牌
    'DISCARD' : 6,
    # 每个位置的副露
    'HALF_FLUSH' : 22,
    'CHI':22,
    'PENG':38,
    'GANG':54,
    # 向听数，用one-hot表示
    'SHANTEN':70
}
```

## 上手指南

```sh
# 先预处理
python3 preprocess.py
# 可以选用数据增强（万、条、筒互换，增加为原来的6倍）
python3 data_augment.py
# 训练监督学习模型
python3 supervised.py #注意命令行参数
# 训练强化学习模型，注意使用监督学习模型参数初始化
python3 train.py
```

