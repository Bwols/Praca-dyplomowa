import torch.nn as nn



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_channels = 3
        self.channel_m = 32
        self.drop_p = 0.3

        self.net = None
        self.load_net_1()

    def forward(self, input):
        return self.net(input)

    def load_net_1(self):
        self.net = nn.Sequential(

            nn.Conv2d(self.input_channels, self.channel_m, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_m, self.channel_m * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_m * 2, self.channel_m * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_m * 4, self.channel_m * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channel_m * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )




    def load_net_3(self):
        self.net = nn.Sequential(

            nn.Conv2d(self.input_channels, self.channel_m, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(self.drop_p),
            nn.Conv2d(self.channel_m, self.channel_m * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(self.drop_p),
            nn.Conv2d(self.channel_m * 2, self.channel_m * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(self.drop_p),
            nn.Conv2d(self.channel_m * 4, self.channel_m * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout(self.drop_p),
            nn.Conv2d(self.channel_m * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )