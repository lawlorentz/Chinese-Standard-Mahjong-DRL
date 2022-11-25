import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self._conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        return x + self._conv(x)


class Bottleneck1D(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        nn.Module.__init__(self)
        self._conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return x + self._conv(x)


class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.conv_up = nn.Sequential(
            nn.Conv2d(70, 128, 3, 1, padding=(1, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0, bias=False),
        )
        self.conv_down = nn.Sequential(
            nn.Conv1d(70, 128, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1, 1, 0, bias=False),
        )

        self.body_up = nn.Sequential(
            *(Bottleneck(128) for _ in range(4)),
        )
        self.body_down = nn.Sequential(
            *(Bottleneck1D(128, 128, 128) for _ in range(4)),
        )
        self.body_total = nn.Sequential(
            *(Bottleneck(128) for _ in range(8)),
        )
        # self.foot = nn.Sequential(
        #     nn.Linear(128*4*7+1, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 235)
        # )
        self._logits = nn.Sequential(
            nn.Linear(128*4*7+1, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(128*4*7+1, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        x = obs
        x_up = x[:, :-1, :3, :]
        x_down = x[:, :-1, 3, :7]
        x_xiangting = torch.squeeze(x[:, -1, 0, :7])
        # ???? x_xiangting=torch.nonzero(x_xiangting)[:,1]
        x_xiangting = torch.nonzero(x_xiangting)[:, 0]
        x_xiangting = x_xiangting.unsqueeze(1)

        x_up = self.conv_up(x_up)
        x_down = self.conv_down(torch.squeeze(x_down, 2))

        x_up = self.body_up(x_up)
        x_down = self.body_down(x_down)

        x_down = torch.unsqueeze(x_down, 2)
        x = torch.cat([x_up, x_down], dim=2)

        x = self.body_total(x)
        x = nn.Flatten()(x)
        x = torch.cat((x, x_xiangting), 1)

        logits = self._logits(x)
        value = self._value_branch(x)

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        masked_logits = logits + inf_mask

        return masked_logits, value
