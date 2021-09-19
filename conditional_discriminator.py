import torch.nn as nn
from torch import cat


def add_image_mask(images,masks):
    x = cat((images, masks), 1)
    return x


class CondDiscriminator(nn.Module):
    def __init__(self):
        self.nc = 4
        self.channels_no = 32 #TODO
        super(CondDiscriminator, self).__init__()
        self.main = None
        self.drop_p = 0.3
        self.load_net1()
        

    def forward(self, image,mask):

        x = add_image_mask(image,mask)
        return self.main(x)
    
    
    def load_net1(self):
        self.main = nn.Sequential(

            nn.Conv2d(self.nc, self.channels_no, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channels_no, self.channels_no * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channels_no * 2, self.channels_no * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channels_no * 4, self.channels_no * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.channels_no * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def load_net2(self):
        self.main = nn.Sequential(

            nn.Conv2d(self.nc, self.channels_no, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.drop_p),

            nn.Conv2d(self.channels_no, self.channels_no * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.drop_p),

            nn.Conv2d(self.channels_no * 2, self.channels_no * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.drop_p),

            nn.Conv2d(self.channels_no * 4, self.channels_no * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.drop_p),

            nn.Conv2d(self.channels_no * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

