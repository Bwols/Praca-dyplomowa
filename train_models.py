from generative_adversarial_nets import GAN
from VAE import VAE
from cond_vae import CondVAE

from fire_mask_dataset import DataLoader, IMAGE_DIR, MASK_DIR

import numpy as np
import torchvision
import time
import os
import torch
from scipy.interpolate import interp1d

from constants import *


def create_interpolate_vectors(vector1, vector2, n ):

    #vector2 = np.array(vector2)
    #vector1 = np.array(vector1)
    distance = vector2 - vector1
    step = distance/(n + 1)

    interpolated_vectors = vector1
    for i in range(n):
        new_inter_vec = vector1 + (step * (i+1))
        interpolated_vectors = torch.cat((interpolated_vectors,new_inter_vec),0)

    #interpolated_vectors.append(vector2)
    interpolated_vectors = torch.cat((interpolated_vectors, vector2), 0)
    return interpolated_vectors



class TrainModels:

    def __init__(self, device, visuals_dir, model_dir, model_type):  # MODEL TYPES CAN BE GAN CGAN,
        self.device = device  # set to use cpu or cuda
        self.model_type = model_type

        self.model_dir = None
        self.visuals_dir = None
        self.set_output_options(visuals_dir, model_dir)

        self.batch_size = 16

        self.train_loader = None
        self.load_train_loader()

        self.model = None
        self.load_model()

        self.loss_criterion = None
        self.choose_loss_criterion()

        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.vae_optimizer = None

        self.choose_optimizer()


        self.CGAN_noide_dim = (1, 8, 8)


    def load_train_loader(self):

        if self.model_type == CGAN or self.model_type == CVAE:
            self.train_loader = DataLoader(IMAGE_DIR, MASK_DIR, self.batch_size, shuffle=True).get_data_loader()
        else:
            self.train_loader = DataLoader(IMAGE_DIR, None, self.batch_size, shuffle=True).get_data_loader()

    def load_model(self):
        if self.model_type == VANILLA_GAN:
            self.model =  GAN(device=self.device,conditonal_GAN=False)
        elif self.model_type == CGAN:
            self.model = GAN(device=self.device,conditonal_GAN=True)

        elif self.model_type == BASIC_VAE:
            self.model = VAE().to(self.device)

        elif self.model_type == CVAE:
            self.model = CondVAE().to(self.device)

    def choose_loss_criterion(self):  # TODO
        self.loss_criterion = torch.nn.BCELoss()


    def set_generator_optimizer(self):
        pass

    def set_discriminator_optimizer(self):
        pass

    def choose_gen_dis_optimizer(self):
        beta1 = 0.5
        self.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.0001, betas=(beta1, 0.999))  # 1 seria 0.001 2 seria 0.0002
        self.generator_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=0.0001, betas=(beta1, 0.999))


    def choose_vae_optimizer(self):
        self.vae_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def choose_optimizer(self):
        if self.model_type == VANILLA_GAN or self.model_type == CGAN:
            self.choose_gen_dis_optimizer()
        else:
            self.choose_vae_optimizer()



    def set_output_options(self, visuals_dir, model_dir):
        make_dir(visuals_dir)
        make_dir(model_dir)

        self.visuals_dir = visuals_dir
        self.model_dir = model_dir

    def save_model(self, epoch):
        if self.model_type == VANILLA_GAN or self.model_type == CGAN:
            generator_path = "{}/gen{}.pth".format(self.model_dir, epoch)
            discriminator_path = "{}/dis{}.pth".format(self.model_dir, epoch)
            self.model.save_models(generator_path=generator_path, discriminator_path= discriminator_path)
        else:
            vae_path = "{}/vae{}.pth".format(self.model_dir, epoch)
            torch.save(self.model.state_dict(), vae_path)


    def log_errors(self):
        pass

    def train_model(self, data):

        if self.model_type == CGAN:
            return self.train_CGAN(data)

        elif self.model_type == VANILLA_GAN:
            return self.train_VANILLA_GAN(data)

        elif self.model_type == BASIC_VAE:
            return self.train_VAE(data)

        elif self.model_type == CVAE:
            return self.train_CVAE(data)

    def train_VANILLA_GAN(self,data):

        self.discriminator_optimizer.zero_grad()

        real_images = data.to(self.device)
        input = create_latent_vector(self.batch_size, VANILLA_GAN_Z).to(self.device)
        fake_images = self.model.generator(input)
        fake_outputs = self.model.discriminator(fake_images)
        real_outputs = self.model.discriminator(real_images)

        dis_loss1 = self.loss_criterion(real_outputs, self.real_labels)
        dis_loss2 = self.loss_criterion(fake_outputs, self.fake_labels)

        dis_loss2 += dis_loss1
        dis_loss2.backward()

        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()

        input = create_latent_vector(self.batch_size, VANILLA_GAN_Z).to(self.device)
        fake_images = self.model.generator(input)
        fake_outputs = self.model.discriminator(fake_images)

        loss = self.loss_criterion(fake_outputs, self.real_labels)
        loss.backward()
        self.generator_optimizer.step()

        return loss.item(),dis_loss2.item()

    def train_CGAN(self,data):

        real_images, masks = data
        real_images = real_images.to(self.device)
        masks = masks.to(self.device)
        z = create_latent_vector(self.batch_size,(1, 8, 8)).to(self.device)
        fake_images = self.model.generator(z, masks)

        self.discriminator_optimizer.zero_grad()


        real_outputs = self.model.discriminator(real_images, masks)

        fake_outputs = self.model.discriminator(fake_images, masks)

        d_x = self.loss_criterion(real_outputs, self.real_labels)
        d_g_z = self.loss_criterion(fake_outputs, self.fake_labels)

        d_x.backward()
        d_g_z.backward()

        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()

        z = create_latent_vector(self.batch_size, self. CGAN_noide_dim).to(self.device)

        fake_images = self.model.generator(z, masks)
        outputs = self.model.discriminator(real_images, masks)

        loss = self.loss_criterion(outputs,self.real_labels)
        loss.backward()
        self.generator_optimizer.step()




        return loss.item(), d_x.item()+ d_g_z.item()


    def train_VAE(self, data):
        self.vae_optimizer.zero_grad()
        real_images = data.to(self.device)
        fake_images, mu, log_var = self.model(real_images)
        reconstruction_loss = self.loss_criterion(fake_images, real_images)
        loss = self.model.final_loss(reconstruction_loss, mu, log_var)
        loss.backward()

        self.vae_optimizer.step()

        return loss,1


    def train_CVAE(self,data):
        self.vae_optimizer.zero_grad()
        real_images, real_masks = data
        real_images = real_images.to(self.device)
        real_masks = real_masks.to(self.device)
        fake_images, mu, log_var = self.model(real_images, real_masks)
        reconstruction_loss = self.loss_criterion(fake_images, real_images)
        loss = self.model.final_loss(reconstruction_loss, mu, log_var)
        loss.backward()

        self.vae_optimizer.step()

        return loss,1



    def generate_fake_images(self, example_input):
        if self.model_type == VANILLA_GAN or self.model_type == CGAN:
            return self.model.generator(example_input)
        else:
            return self.model.decode(example_input)







    def train_loop(self, epochs):
        start = time.time()

        self.real_labels = torch.ones(self.batch_size, 1, 1, 1).to(self.device)
        self.fake_labels = torch.zeros(self.batch_size, 1, 1, 1).to(self.device)
        example_input = create_example_input(16)


        for epoch in range(epochs):
            gen_run_loss = 0
            dis_run_loss = 0

            print("[{}] epoch".format(epoch))
            epoch_start = time.time()
            for i, data in enumerate(self.train_loader):
                errors = self.train_model(data)
                gen_loss, diss_loss = errors
                gen_run_loss += gen_loss
                dis_run_loss += diss_loss


            gen_run_loss = gen_run_loss #######            -<<<<<<<< tu do zmiany #TODO
            dis_run_loss = dis_run_loss

            gen_run_loss /= 207 *self.batch_size
            dis_run_loss /= 207 *self.batch_size
            print("     generator mean loss: {}".format(gen_run_loss))
            print("     discriminator mean loss: {}".format(dis_run_loss))



            epoch_end = time.time()

            example_fake_images = self.generate_fake_images(example_input)# trzeba zmienić argumenty w GANIE
            save_image_batch(self.visuals_dir,"{}".format(epoch),example_fake_images)

            self.save_model(epoch)

            print("     time of epoch: {}".format(epoch_end-epoch_start))

        end = time.time()
        exec_time = end - start
        print("time of training :{} minutes".format(exec_time/60))




