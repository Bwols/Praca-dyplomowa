import torch.nn as nn

from constants import NC
from torch import exp, rand_like, sum

from cond_vae import CondVAEEncoderDecoder
from basic_vae import VAEEncoderDecoder

import torch
from constants import  create_latent_vector


class VAE:
    def __init__(self, device, conditionalVAE = False):
        self.device = device
        self.conditionalVAE = conditionalVAE

        self.optimizer = None
        self.loss_criterion = None

        self.encoder_decoder = None
        self.choose_encoder_decoder()

    def choose_encoder_decoder(self):
        if self.conditionalVAE:

            self.encoder_decoder = CondVAEEncoderDecoder().to(self.device)
        else:
            self.encoder_decoder = VAEEncoderDecoder().to(self.device)

    def set_model_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_criterion(self, loss_criterion):
        self.loss_criterion = loss_criterion

    def get_parameters(self):
        return self.encoder_decoder.parameters()

    def generate(self, data):
        return self.encoder_decoder.decode(data)

    def encode_decode(self, data):
        if self.conditionalVAE :
            return self.encoder_decoder.forward(*data)
        else:
            return self.encoder_decoder.forward(data)

    def train(self, data):
        if self.conditionalVAE:
            return self.train_conditional(data)
        else:
            return self.train_basic(data)

    def load_encoder_decoder(self, encoder_decoder_path):
        self.encoder_decoder.load_state_dict(torch.load(encoder_decoder_path))


    def train_basic(self, data):
        self.optimizer.zero_grad()

        real_images = data.to(self.device)

        fake_images, mu, log_var = self.encoder_decoder(real_images)
        reconstruction_loss = self.loss_criterion(fake_images, real_images)
        loss = self.encoder_decoder.final_loss(reconstruction_loss, mu, log_var)

        loss.backward()

        self.optimizer.step()

        return loss

    def train_basic2(self, data):
        self.optimizer.zero_grad()

        real_images = data.to(self.device)
        fake_images, mu, log_var = self.encoder_decoder(real_images)

        z = create_latent_vector(real_images.shape[0], (100, 1, 1)).to(self.device)
        fake_images2 = self.generate(z)

        loss_criterion2 = torch.nn.BCELoss(reduction="sum")
        loss2 = loss_criterion2(fake_images2, real_images)


        reconstruction_loss = self.loss_criterion(fake_images, real_images)
        loss = self.encoder_decoder.final_loss(reconstruction_loss +loss2, mu, log_var)

        #loss.backward()




        loss.backward()
        self.optimizer.step()

        return loss


    def train_conditional(self, data):
        self.optimizer.zero_grad()
        real_images, real_masks = data
        real_images = real_images.to(self.device)
        real_masks = real_masks.to(self.device)
        fake_images, mu, log_var = self.encoder_decoder(real_images, real_masks)
        reconstruction_loss = self.loss_criterion(fake_images, real_images)
        loss = self.encoder_decoder.final_loss(reconstruction_loss, mu, log_var)
        loss.backward()

        self.optimizer.step()

        return loss