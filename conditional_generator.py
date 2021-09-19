import torch.nn as nn
from torch import cat


class CondGenerator_UNET(nn.Module):

    def __init__(self):
        super(CondGenerator_UNET,self).__init__()
        self.latent_map_channels = 64
        self.map_channels = [64, 128, 256]
        self.drop_p = 0.1

        self.downscale1 = self.down_part(1, self.map_channels[0], 3, 1)
        self.downscale2 = self.down_part(self.map_channels[0], self.map_channels[1], 3, 1)
        self.downscale3 = self.down_part(self.map_channels[1], self.map_channels[2] - self.latent_map_channels, 3, 1)

        self.upscale3 = self.up_part(self.map_channels[2],self.map_channels[1],3,1)
        self.upscale2 = self.up_part(self.map_channels[1]*2, self.map_channels[0], 3, 1)
        self.upscale1 = self.up_part(self.map_channels[0] * 2, 3, 3, 1)




        """ 
        self.downscale1 = self.down_part(1, self.map_channels[0], 3, 1)
        self.downscale2 = self.down_part(self.map_channels[0], self.map_channels[1], 3, 1)
        self.downscale3 = self.down_part(self.map_channels[1], self.map_channels[2] - self.latent_map_channels, 3, 1)

        self.upscale3 = self.up_part(self.map_channels[2],self.map_channels[1],3,1)
        self.upscale2 = self.up_part(self.map_channels[1]*2, self.map_channels[0], 3, 1)
        self.upscale1 = self.up_part(self.map_channels[0] * 2, 3, 3, 1)
        """

        self.convolute_latent = self.latent_transformation_layer()



    """
    input:latent vector BATCH_SIZEx1x8x8 mask BATCH_SIZEx3x64x64 
    """


    def forward(self,latent, masks):#masks size 64

        masks_32 = self.downscale1(masks)#64 channels
        masks_16 = self.downscale2(masks_32)#128
        masks_8 = self.downscale3(masks_16)#512


        latent_8 = self.convolute_latent(latent)
        x_8 = self.add_tensors(latent_8,masks_8)

        x_16 = self.upscale3(x_8)
        x_32 = self.upscale2(self.add_tensors(x_16,masks_16))
        x_64 = self.upscale1(self.add_tensors(x_32,masks_32))

        return x_64

    def down_part(self,in_channels,out_channels,kernel_size,padding):
        one_down = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)# downsizes by 2
        )
        return one_down


    def down_part_2(self,in_channels,out_channels,kernel_size,padding):
        one_down = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=1,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.drop_p),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)# downsizes by 2
        )
        return one_down





    def latent_transformation_layer(self):
        layer = nn.Sequential(
            nn.Conv2d(1,self.latent_map_channels,kernel_size=3,stride=1,padding=1)
        )
        return layer

    def add_tensors(self,tensor1,tensor2):
        return cat((tensor1,tensor2),1)

    def up_part(self,in_channels,out_channels,kernel_size,padding):
        one_up = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                        output_padding=1)
                               )
        return one_up


    def up_part_2(self,in_channels,out_channels,kernel_size,padding):
        one_up = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Dropout(self.drop_p),
                               nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Dropout(self.drop_p),
                               nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                        output_padding=1)
                               )
        return one_up



def add_latent_mask(latent, mask):  # dim(1,100) i 1(64,64)
    mask = mask.view(mask.size()[0],-1)
    x = cat((latent,mask),1)
    return x


from torch import cat
class CondGenerator(nn.Module):

    def __init__(self):
        super(CondGenerator, self).__init__()
        self.nz = 4196
        self.nc = 3
        self.channels_no = 32 #32 before TODO -
        self.drop_p = 0.3
        self.main = None
        self.load_net1()

    def forward(self,latent,mask ):
        #print(mask.size())
        x = add_latent_mask(latent,mask)
        x = x.view(x.size()[0],x.size()[1],1,1)
        #print("forward",x.size())
        return self.main(x)

    def load_net1(self):
        self.main = nn.Sequential(

            nn.ConvTranspose2d(self.nz, self.channels_no * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels_no * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channels_no * 8, self.channels_no * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channels_no * 4, self.channels_no * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channels_no * 2, self.channels_no, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channels_no, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )


    def load_net2(self):
        self.main = nn.Sequential(

            nn.ConvTranspose2d(self.nz, self.channels_no * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channels_no * 8),
            nn.ReLU(True),
            nn.Dropout(self.drop_p),

            nn.ConvTranspose2d(self.channels_no * 8, self.channels_no * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 4),
            nn.ReLU(True),
            nn.Dropout(self.drop_p),

            nn.ConvTranspose2d(self.channels_no * 4, self.channels_no * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no * 2),
            nn.ReLU(True),
            nn.Dropout(self.drop_p),

            nn.ConvTranspose2d(self.channels_no * 2, self.channels_no, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channels_no),
            nn.ReLU(True),
            nn.Dropout(self.drop_p),

            nn.ConvTranspose2d(self.channels_no, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )



