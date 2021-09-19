import torch.nn as nn


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.noise_z = 100
        self.channel_m = 32
        self.output_channels = 3

        self.drop_p = 0.3

        self.net = None
        self.load_net_3()

    def forward(self, input):
        return self.net(input)

    def load_net_1(self):
        self.net  = nn.Sequential(

            nn.ConvTranspose2d(self.noise_z, self.channel_m * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_m * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 8, self.channel_m * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 4, self.channel_m * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 2, self.channel_m, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m, self.output_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )


    def load_net_4(self):
        self.net  = nn.Sequential(

            nn.ConvTranspose2d(self.noise_z, self.channel_m * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_m * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 8, self.channel_m * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 4),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode="bilinear"),

            nn.ConvTranspose2d(self.channel_m * 4, self.channel_m, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m, self.output_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )



    def load_net_2(self):
        self.net  = nn.Sequential(
            nn.ConvTranspose2d(self.noise_z, self.channel_m * 16, 4, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 16, self.channel_m * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 8, self.channel_m * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 4, self.channel_m * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m * 2, self.channel_m, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m , self.channel_m, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_m),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_m, self.channel_m, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_m),
            nn.ReLU(True),


            nn.ConvTranspose2d(self.channel_m, self.output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )



    def load_net_3(self):
        self.net  = nn.Sequential(

            nn.ConvTranspose2d(self.noise_z, self.channel_m * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_m * 8),
            nn.ReLU(True),

            nn.Dropout(self.drop_p),
            nn.ConvTranspose2d(self.channel_m * 8, self.channel_m * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 4),
            nn.ReLU(True),

            nn.Dropout(self.drop_p),
            nn.ConvTranspose2d(self.channel_m * 4, self.channel_m * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m * 2),
            nn.ReLU(True),

            nn.Dropout(self.drop_p),
            nn.ConvTranspose2d(self.channel_m * 2, self.channel_m, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_m),
            nn.ReLU(True),

            nn.Dropout(self.drop_p),
            nn.ConvTranspose2d(self.channel_m, self.output_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )
"""
import torch
latent_noise = torch.rand(16,100,1,1)
generator = Generator()
x = generator(latent_noise)
print(x.shape[:])
"""
