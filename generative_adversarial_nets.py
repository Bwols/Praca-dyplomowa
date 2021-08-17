import torch
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator

from conditional_generator import CondGenerator_UNET








def add_image_mask(images,masks):
    x = torch.cat((images, masks), 1)
    return x


class CondDiscriminator(nn.Module):
    def __init__(self):
        nc = 4
        ndf = 64
        super(CondDiscriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image,mask):
        x = add_image_mask(image,mask)
        return self.main(x)







class GAN:
    def __init__(self, device,conditonal_GAN = False):
        self.device = device

        self.generator = None
        self.choose_generator(conditonal_GAN)

        self.discriminator = None
        self.choose_discrminator(conditonal_GAN)

        self.discriminator_optimizer = None
        self.generator_optimizer = None


    def choose_generator(self, conditional_GAN):
        if conditional_GAN:
            self.generator = CondGenerator_UNET().to(self.device)
        else:
            self.generator = Generator().to(self.device)

    def choose_discrminator(self, conditional_GAN):
        if conditional_GAN:
            self.discriminator = CondDiscriminator().to(self.device)
        else:
            self.discriminator = Discriminator().to(self.device)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def pass_through_network(self, input):
        fake_image = self.generator(input)
        fake_output = self.discriminator(fake_image)
        return (fake_image, fake_output)





    def set_gen_dis_optimizer(self):
        beta1 = 0.5

    def save_models(self, generator_path, discriminator_path):
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def load_models(self, generator_path, discriminator_path):
        pass

    def load_generator(self, generator_path):
        self.generator.load_state_dict(torch.load(generator_path))


