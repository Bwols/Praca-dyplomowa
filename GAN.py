import torch
import torch.nn as nn
from constants import  *

from generator import Generator
from discriminator import Discriminator

from conditional_generator import CondGenerator_UNET , CondGenerator
from conditional_discriminator import CondDiscriminator



class GAN:
    def __init__(self, device,conditonal_GAN = False):
        self.device = device
        self.conditional_GAN = conditonal_GAN

        self.generator = None
        self.choose_generator()

        self.discriminator = None
        self.choose_discrminator()

        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.loss_criterion = None

        self.gen_advantage_k = 4


    def choose_generator(self):
        if self.conditional_GAN:
            #self.generator = CondGenerator_UNET().to(self.device)
            self.generator = CondGenerator().to(self.device)
        else:
            self.generator = Generator().to(self.device)

    def choose_discrminator(self):
        if self.conditional_GAN:
            self.discriminator = CondDiscriminator().to(self.device)
            #self.discriminator = Discriminator().to(self.device)  # TODO tutaj usuanc
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
        if not self.conditional_GAN:
            return  self.train_vanilla_gen_loop_k(data)
            #return  self.train_vanilla(data)
        else:
            #return self.train_conditional_beta(data) # TODO tu też usunąć
            return self.train_conditional(data)


    def train_vanilla(self, data):

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

    def train_vanilla_gen_loop_k(self, data):
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
        gen_loss_k = 0
        for i in range(self.gen_advantage_k):
            self.generator_optimizer.zero_grad()
            input = create_latent_vector(batch_size, VANILLA_GAN_Z).to(self.device)
            fake_images = self.generator(input)
            fake_outputs = self.discriminator(fake_images)

            gen_loss = self.loss_criterion(fake_outputs, real_labels)
            gen_loss.backward()
            self.generator_optimizer.step()
            gen_loss_k+=gen_loss

        return gen_loss_k.item()/self.gen_advantage_k, dis_loss2.item()

    def train_conditional(self, data):
        real_images, masks = data
        batch_size = real_images.shape[0]
        real_images = real_images.to(self.device)
        masks = masks.to(self.device)

        z = create_latent_vector(batch_size, CGAN_Z).to(self.device)
        fake_images = self.generator(z, masks)

        self.discriminator_optimizer.zero_grad()

        real_outputs = self.discriminator(real_images, masks)

        fake_outputs = self.discriminator(fake_images, masks)

        real_labels, fake_labels = create_labels(batch_size, self.device)
        d_x = self.loss_criterion(real_outputs, real_labels)
        d_g_z = self.loss_criterion(fake_outputs, fake_labels)

        d_x.backward()
        d_g_z.backward()

        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()

        z = create_latent_vector(batch_size, CGAN_Z).to(self.device)

        fake_images = self.generator(z, masks)
        outputs = self.discriminator(fake_images, masks)

        loss = self.loss_criterion(outputs, real_labels)
        loss.backward()
        self.generator_optimizer.step()

        return loss.item(), d_x.item() + d_g_z.item()



    def train_conditional_beta(self, data):

        real_images, masks = data
        batch_size = real_images.shape[0]
        real_images = real_images.to(self.device)
        masks = masks.to(self.device)

        z = create_latent_vector(batch_size, CGAN_Z).to(self.device)
        fake_images = self.generator(z, masks)

        self.discriminator_optimizer.zero_grad()

        real_outputs = self.discriminator(real_images)

        fake_outputs = self.discriminator(fake_images)

        real_labels, fake_labels = create_labels(batch_size, self.device)
        d_x = self.loss_criterion(real_outputs, real_labels)
        d_g_z = self.loss_criterion(fake_outputs, fake_labels)

        d_x.backward()
        d_g_z.backward()

        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()

        fake_images = self.generator(z, masks)
        outputs = self.discriminator(fake_images)

        fake_masks = create_white_mask_tensor(fake_images)
        fake_masks = fake_masks.to(self.device)

        mask_loss_criterion = torch.nn.MSELoss(reduction='mean')#set to sum
        mask_loss = mask_loss_criterion(fake_masks, masks)
        loss = self.loss_criterion(outputs, real_labels) + mask_loss
        loss.backward()
        self.generator_optimizer.step()

        return loss.item(), d_x.item() + d_g_z.item()



class CGAN:
    def __init__(self, device):
        self.device = device

        self.generator = CondGenerator_UNET().to(self.device)

        self.discriminator = CondDiscriminator().to(self.device)

        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.loss_criterion = None

        def set_generator_optimizer(self, generator_optimizer):
            self.generator_optimizer = generator_optimizer

        def set_discriminator_optimizer(self, discriminator_optimizer):
            self.discriminator_optimizer = discriminator_optimizer

        def set_loss_criterion(self, loss_critertion):
            self.loss_criterion = loss_critertion

        def save_model(self, generator_path, discriminator_path):
            torch.save(self.generator.state_dict(), generator_path)
            torch.save(self.discriminator.state_dict(), discriminator_path)

        def train():
            pass