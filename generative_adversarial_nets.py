import torch
import torch.nn as nn
from constants import  *

from generator import Generator
from discriminator import Discriminator

from conditional_generator import CondGenerator_UNET
from conditional_discriminator import CondDiscriminator



class GAN:
    def __init__(self, device,conditonal_GAN = False):
        self.device = device

        self.generator = None
        self.choose_generator(conditonal_GAN)

        self.discriminator = None
        self.choose_discrminator(conditonal_GAN)

        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.loss_criterion = None




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



    def set_generator_optimizer(self, generator_optimizer):
        self.generator_optimizer = generator_optimizer

    def set_discriminator_optimizer(self, discriminator_optimizer):
        self.discriminator_optimizer = discriminator_optimizer

    def set_loss_criterion(self, loss_critertion):
        self.loss_criterion = loss_critertion



    def save_model(self, generator_path, discriminator_path):
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def load_models(self, generator_path, discriminator_path):
        pass

    def load_generator(self, generator_path):
        self.generator.load_state_dict(torch.load(generator_path))

    def train(self, data):

        self.discriminator_optimizer.zero_grad()


        real_images = data.to(self.device)
        batch_size = real_images.shape[0]
        input = create_latent_vector(batch_size, VANILLA_GAN_Z).to(self.device)

        fake_images = self.generator(input)
        fake_outputs = self.discriminator(fake_images)
        real_outputs = self.discriminator(real_images)

        real_labels, fake_labels = create_labels(batch_size, self.device)
        dis_loss1 = self.loss_criterion(real_outputs, real_labels)
        dis_loss2 = self.loss_criterion(fake_outputs, fake_labels)

        dis_loss2 += dis_loss1
        dis_loss2.backward()

        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()

        input = create_latent_vector(batch_size, VANILLA_GAN_Z).to(self.device)
        fake_images = self.generator(input)
        fake_outputs = self.discriminator(fake_images)

        loss = self.loss_criterion(fake_outputs, real_labels)
        loss.backward()
        self.generator_optimizer.step()

        return loss.item(), dis_loss2.item()


