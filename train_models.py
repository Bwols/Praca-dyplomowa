from generative_adversarial_nets import GAN
from VAE import VAE
#from cond_vae import CondVAE

import visualize_results as vis_rez
from fire_mask_dataset import DataLoader, IMAGE_DIR, MASK_DIR
import torch.nn as nn
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

    def __init__(self, device, output_dir, model_type):  # MODEL TYPES CAN BE GAN CGAN,
        self.device = device  # set to use cpu or cuda
        self.model_type = model_type

        self.options_set = {}

        self.output_dir = output_dir
        self.model_dir = "{}/models".format(output_dir)
        self.visuals_dir = "{}/visuals".format(output_dir)
        self.set_output_options()

        self.batch_size = 16

        self.train_loader = None
        self.load_train_loader()

        self.model = None
        self.load_model()

        self.loss_function_name = BCE
        self.loss_criterion = torch.nn.BCELoss()
        self.model.set_loss_criterion(self.loss_criterion)

        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.vae_optimizer = None







    def load_train_loader(self):

        if self.model_type == CGAN or self.model_type == CVAE:
            self.train_loader = DataLoader(IMAGE_DIR, MASK_DIR, self.batch_size, shuffle=True).get_data_loader()
        else:
            self.train_loader = DataLoader(IMAGE_DIR, None, self.batch_size, shuffle=True).get_data_loader()

    def load_model(self):

        if self.model_type == VANILLA_GAN:
            self.model = GAN(device=self.device, conditonal_GAN=False)
        elif self.model_type == CGAN:
            self.model = GAN(device=self.device, conditonal_GAN=True)

        elif self.model_type == BASIC_VAE:
            self.model = VAE(device=self.device, conditionalVAE=False)

        elif self.model_type == CVAE:
            self.model = VAE(device=self.device, conditionalVAE=True)

    def set_loss_criterion(self, loss_function_name):

        self.loss_function_name = loss_function_name
        if loss_function_name == L1:
            self.loss_criterion = torch.nn.L1Loss()
        elif loss_function_name ==  MSE:
            self.loss_criterion = torch.nn.MSELoss()
        elif loss_function_name == NEGATIVE_LOG_LIKELIHOOD:
            self.loss_criterion = torch.nn.NLLLoss()
        elif loss_function_name == CROSS_ENTROPY_LOSS:
            self.loss_criterion = torch.nn.CrossEntropyLoss()
        elif loss_function_name == KLD:
            self.loss_criterion = torch.nn.KLDivLoss()
        elif loss_function_name == BCE:
            self.loss_criterion = torch.nn.BCELoss()
        elif loss_function_name == HINGE_EMBEDDING_LOSS:
            self.loss_criterion = torch.nn.HingeEmbeddingLoss()

        self.model.set_loss_criterion(self.loss_criterion)



    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        pass

    def set_net_optimizer(self, net_name, optimizer_name, lr ,momentum): #momentum is beta for adam and admaw
        parameters = None
        optimizer = None

        if net_name == GENERATOR:
            parameters = self.model.generator.parameters()
        elif net_name == DISCRIMINATOR:
            parameters = self.model.discriminator.parameters()
        elif net_name == VAE_NET:
            parameters = self.model.encoder_decoder.parameters()



        if optimizer_name == ADAM:
            optimizer = torch.optim.Adam(parameters, lr=lr, betas=(momentum, 0.999))

        elif optimizer_name == ADAMW:
            optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(momentum, 0.999))

        elif optimizer_name == SGD:
            optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)

        if net_name == GENERATOR:
            self.generator_optimizer = optimizer
            self.model.set_generator_optimizer(optimizer)
        elif net_name == DISCRIMINATOR:
            self.discriminator_optimizer = optimizer
            self.model.set_discriminator_optimizer(optimizer)
        elif net_name == VAE_NET:
            self.vae_optimizer = optimizer
            self.model.set_model_optimizer(optimizer)










    def set_output_options(self):
        make_dir(self.output_dir)
        make_dir(self.visuals_dir)
        make_dir(self.model_dir)


    def save_model(self, epoch):
        if self.model_type == VANILLA_GAN or self.model_type == CGAN:
            generator_path = "{}/gen{}.pth".format(self.model_dir, epoch)
            discriminator_path = "{}/dis{}.pth".format(self.model_dir, epoch)
            self.model.save_model(generator_path=generator_path, discriminator_path=discriminator_path)
        else:
            vae_path = "{}/vae{}.pth".format(self.model_dir, epoch)
            torch.save(self.model.encoder_decoder.state_dict(), vae_path)




    def train_model(self, data):
        return self.model.train(data)



    def generate_fake_images(self, example_input):
        if self.model_type == VANILLA_GAN:
            return self.model.generator(example_input)
        elif self.model_type == CGAN:
            return self.model.generator(*example_input)
        else:
            return self.model.generate(example_input)



    class MeasureLogTraining:

        def __init__(self, model_type, output_dir):

            self.plot_dir = "{}/plots".format(output_dir)
            make_dir(self.plot_dir)

            self.model_type = model_type

            self.gen_loss_v = []
            self.dis_loss_v = []
            self.vae_loss_v = []

            self.gen_epoch_loss = 0
            self.dis_epoch_loss = 0
            self.vae_epoch_loss = 0

            self.start_t = 0

        def start_timer(self):
            self.start_t = time.time()

        def stop_timer(self):
            self.stop_t = time.time()
            execution_time = (self.stop_t - self.start_t) / 60  # In minutes
            print("Final Time of training :{} minutes".format(execution_time))

        def reset_epoch_loss(self):
            self.gen_epoch_loss = 0
            self.dis_epoch_loss = 0
            self.vae_epoch_loss = 0


        def add_batch_loss(self, model_batch_loss):

            if self.model_type == CGAN or self.model_type == VANILLA_GAN:
                gen_batch_loss, dis_batch_loss = model_batch_loss
                self.gen_epoch_loss += gen_batch_loss
                self.dis_epoch_loss += dis_batch_loss

            else:
                self.vae_epoch_loss += model_batch_loss

        def append_epoch_loss(self):

            if self.model_type == CGAN or self.model_type == VANILLA_GAN: #gen_run_loss /= 207 *self.batch_size
                self.gen_loss_v.append(self.gen_epoch_loss)
                self.dis_loss_v.append(self.dis_epoch_loss)

                print("     Generator epoch loss: {}".format(self.gen_epoch_loss))
                print("     Discriminator epoch loss: {}".format(self.dis_epoch_loss))


            elif self.model_type == BASIC_VAE or self.model_type == CVAE:
                self.vae_loss_v.append(self.vae_epoch_loss)
                print("     VAE epoch loss: {}".format(self.vae_epoch_loss))


        def save_plot(self):

            if self.model_type == CGAN or self.model_type == VANILLA_GAN:
                title = "Błędy generatora i dyskryminatora w procesie nauki"
                plot_path = "{}/gen_diss_loss.png".format(self.plot_dir)
                vis_rez.draw_gen_dis_plot(self.gen_loss_v, self.dis_loss_v, title, plot_path)

                gen_file_path = "{}/gen_loss.txt".format(self.plot_dir)
                vis_rez.save_model_loss_data(self.gen_loss_v,gen_file_path)

                dis_file_path = "{}/dis_loss.txt".format(self.plot_dir)
                vis_rez.save_model_loss_data(self.dis_loss_v, dis_file_path)

            elif self.model_type == BASIC_VAE or self.model_type == CVAE:
                title = "Błąd autoenkodera wariacyjnego w procesie nauki"
                plot_path = "{}/vae_loss.png".format(self.plot_dir)

                vis_rez.draw_vae_loss_plot(self.vae_loss_v, title, plot_path)

                gen_file_path = "{}/gen_loss.txt".format(self.plot_dir)
                vis_rez.save_model_loss_data(self.vae_loss_v, gen_file_path)


    def train_loop(self, epochs):

        measure_log_train = self.MeasureLogTraining(self.model_type, self.output_dir)
        measure_log_train.start_timer()

        self.real_labels = torch.ones(self.batch_size, 1, 1, 1).to(self.device)
        self.fake_labels = torch.zeros(self.batch_size, 1, 1, 1).to(self.device)
        example_input = create_example_input(16, self.model_type, self.device)


        for epoch in range(epochs):
            print("Epoch[{}]:".format(epoch+1))

            measure_log_train.reset_epoch_loss()


            epoch_start = time.time()
            for i, data in enumerate(self.train_loader):
                model_batch_loss = self.train_model(data)

                measure_log_train.add_batch_loss(model_batch_loss)

            epoch_end = time.time()

            measure_log_train.append_epoch_loss()
            measure_log_train.save_plot()


            example_fake_images = self.generate_fake_images(example_input)# trzeba zmienić argumenty w GANIE
            save_image_batch(self.visuals_dir,"{}".format(epoch),example_fake_images)
            self.save_model(epoch)

            print("     Epoch time: {} s".format(epoch_end-epoch_start))


        measure_log_train.stop_timer()



