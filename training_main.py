import torch
from train_models import TrainModels
from constants import *

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
"""
possible models


testy zrobione na ADAMW to tak naprawdę ADAM
"""
def train_GAN():
    ta = TrainModels(device,"CGAN_23.08_BCE_ADAMW_beta", model_type="CGAN")
    ta.set_loss_criterion("BCE")
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator",ADAMW, lr=0.01, momentum=0.5)
    ta.set_net_optimizer("discriminator",ADAMW, lr=0.01, momentum=0.5)

    ta.train_loop(100)

train_GAN()


ta = TrainModels(device,"CVAE_22.08_ADAMW", model_type="VAE")
ta.set_loss_criterion("BCE")
ta.set_batch_size(16)
ta.set_net_optimizer("VAE", ADAMW, lr=0.01, momentum=0.5)


#ta.train_loop(100)




"""
    def choose_gen_dis_optimizer(self):
        beta1 = 0.5
        self.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.0002, betas=(beta1, 0.999))  # 1 seria 0.001 2 seria 0.0002
        self.generator_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=0.0002, betas=(beta1, 0.999))

        self.model.set_discriminator_optimizer(self.discriminator_optimizer)
        self.model.set_generator_optimizer(self.generator_optimizer)

    def choose_vae_optimizer(self):
        self.vae_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
        def set_generator_optimizer(self, optimizer_name, lr, momentum):
        if optimizer_name == ADAM:
            optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(momentum, 0.999))

        elif optimizer_name == ADAMW:
            optimizer = torch.optim.AdamW(self.model.generator.parameters(),lr=lr,betas=(momentum,0.999))

        elif optimizer_name == SGD:
            optmizer = torch.optim.SGD(self.model.generator.parameters(),lr=lr, momentum= momentum)


     def choose_optimizer(self):
        if self.model_type == VANILLA_GAN or self.model_type == CGAN:
            self.choose_gen_dis_optimizer()
        else:
            self.choose_vae_optimizer()



"""